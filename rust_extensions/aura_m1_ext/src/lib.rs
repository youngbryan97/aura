use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustpython_parser::ast::{self, Constant, ExceptHandler, Expr, Ranged, Stmt};
use rustpython_parser::Parse;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

#[cfg(target_os = "macos")]
use std::os::raw::c_int;

// Extern link to macOS-specific thread QoS APIs.
#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
}

// macOS QoS Class definitions (standard for Apple Silicon)
#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INITIATED: u32 = 0x19; // P-cores (Fast compute)
#[cfg(target_os = "macos")]
const QOS_CLASS_UTILITY: u32 = 0x15; // E-cores (Background IO/Sensory)

const MAX_SKILL_SOURCE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SKILL_ROOTS: usize = 32;
const MAX_TRAVERSAL_ENTRIES: usize = 100_000;
const SKILL_DECORATORS: [&str; 4] = ["aura_skill", "capability_skill", "register_skill", "skill"];
const BASE_SKILL_NAMES: [&str; 3] = [
    "core.skills.base_skill.BaseSkill",
    "infrastructure.BaseSkill",
    "infrastructure.base_skill.BaseSkill",
];
const METADATA_FIELDS: [&str; 11] = [
    "abstract",
    "constructor_dependencies",
    "description",
    "effect_scope",
    "enabled",
    "execution_profile",
    "is_core_personality",
    "memory_mb_estimate",
    "metabolic_cost",
    "name",
    "timeout_seconds",
];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillRootInput {
    path: String,
    package: String,
    kind: String,
}

#[derive(Clone, Debug)]
struct TrustedSkillRoot {
    path: PathBuf,
    package: String,
    kind: String,
}

#[derive(Clone, Debug)]
struct ModuleRecord {
    path: PathBuf,
    module_path: String,
    source_kind: String,
    source_path: String,
    source_sha256: String,
    source: String,
    tree: ast::Suite,
    aliases: BTreeMap<String, String>,
    local_classes: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct ParsedClass {
    module_path: String,
    class_name: String,
    source_kind: String,
    source_path: String,
    source_sha256: String,
    line: usize,
    bases: Vec<String>,
    metadata: BTreeMap<String, Value>,
    metadata_errors: Vec<String>,
    decorated: bool,
    meaningful_body: bool,
}

impl ParsedClass {
    fn qualified_name(&self) -> String {
        format!("{}.{}", self.module_path, self.class_name)
    }
}

#[derive(Clone, Debug, Serialize)]
struct CatalogIssue {
    code: String,
    severity: String,
    detail: String,
    module_path: String,
    class_name: String,
    source_path: String,
    line: usize,
}

impl CatalogIssue {
    fn error(code: &str, detail: impl Into<String>, record: Option<&ParsedClass>) -> Self {
        Self {
            code: code.to_string(),
            severity: "error".to_string(),
            detail: detail.into(),
            module_path: record.map_or_else(String::new, |item| item.module_path.clone()),
            class_name: record.map_or_else(String::new, |item| item.class_name.clone()),
            source_path: record.map_or_else(String::new, |item| item.source_path.clone()),
            line: record.map_or(0, |item| item.line),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct SkillCandidate {
    name: String,
    description: String,
    module_path: String,
    class_name: String,
    source_kind: String,
    source_path: String,
    source_sha256: String,
    line: usize,
    effect_scope: String,
    authority_class: String,
    constructor_dependencies: Vec<String>,
    decorated: bool,
    inherited_metadata: bool,
    exclusion_reason: String,
    catalog_id: String,
}

fn valid_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|item| item == '_' || item.is_ascii_alphanumeric())
}

fn valid_package(value: &str) -> bool {
    !value.is_empty() && value.split('.').all(valid_identifier)
}

fn valid_kind(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic())
        && value.len() <= 64
        && chars.all(|item| item == '_' || item == '-' || item.is_ascii_alphanumeric())
}

fn reject_symlink_components(path: &Path) -> Result<(), String> {
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) => current.push(component.as_os_str()),
            Component::Normal(part) => {
                current.push(part);
                let metadata = fs::symlink_metadata(&current)
                    .map_err(|error| format!("cannot inspect {}: {error}", current.display()))?;
                if metadata.file_type().is_symlink() {
                    return Err(format!(
                        "trusted root contains symlink component: {}",
                        current.display()
                    ));
                }
            }
            Component::CurDir | Component::ParentDir => {
                return Err(format!(
                    "trusted root is not lexically canonical: {}",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

fn parse_trusted_roots(roots_json: &str) -> Result<Vec<TrustedSkillRoot>, String> {
    let inputs: Vec<SkillRootInput> =
        serde_json::from_str(roots_json).map_err(|error| format!("invalid roots JSON: {error}"))?;
    if inputs.is_empty() || inputs.len() > MAX_SKILL_ROOTS {
        return Err(format!(
            "trusted roots must contain 1..={MAX_SKILL_ROOTS} entries"
        ));
    }

    let mut roots = Vec::with_capacity(inputs.len());
    let mut identities = BTreeSet::new();
    for input in inputs {
        if !valid_package(&input.package) {
            return Err(format!("invalid root package: {:?}", input.package));
        }
        if !valid_kind(&input.kind) {
            return Err(format!("invalid root kind: {:?}", input.kind));
        }
        let supplied = PathBuf::from(&input.path);
        if !supplied.is_absolute() {
            return Err(format!(
                "trusted root must be absolute: {}",
                supplied.display()
            ));
        }
        reject_symlink_components(&supplied)?;
        let canonical = supplied
            .canonicalize()
            .map_err(|error| format!("cannot canonicalize {}: {error}", supplied.display()))?;
        if canonical != supplied {
            return Err(format!(
                "trusted root must already be canonical: {}",
                supplied.display()
            ));
        }
        let metadata = fs::metadata(&canonical)
            .map_err(|error| format!("cannot stat {}: {error}", canonical.display()))?;
        if !metadata.is_dir() {
            return Err(format!(
                "trusted root is not a directory: {}",
                canonical.display()
            ));
        }
        let identity = (canonical.clone(), input.package.clone(), input.kind.clone());
        if !identities.insert(identity) {
            continue;
        }
        roots.push(TrustedSkillRoot {
            path: canonical,
            package: input.package,
            kind: input.kind,
        });
    }
    for (index, left) in roots.iter().enumerate() {
        for right in roots.iter().skip(index + 1) {
            if left.path.starts_with(&right.path) || right.path.starts_with(&left.path) {
                return Err(format!(
                    "trusted roots overlap: {} and {}",
                    left.path.display(),
                    right.path.display()
                ));
            }
        }
    }
    roots.sort_by(|left, right| {
        (&left.path, &left.package, &left.kind).cmp(&(&right.path, &right.package, &right.kind))
    });
    Ok(roots)
}

fn collect_python_files(root: &TrustedSkillRoot) -> Result<Vec<PathBuf>, String> {
    let mut pending = vec![root.path.clone()];
    let mut files = Vec::new();
    let mut entries_seen = 0usize;
    while let Some(directory) = pending.pop() {
        let mut entries = fs::read_dir(&directory)
            .map_err(|error| format!("cannot read directory {}: {error}", directory.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!(
                    "cannot enumerate directory {}: {error}",
                    directory.display()
                )
            })?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            entries_seen += 1;
            if entries_seen > MAX_TRAVERSAL_ENTRIES {
                return Err(format!(
                    "trusted root exceeded {MAX_TRAVERSAL_ENTRIES} traversal entries: {}",
                    root.path.display()
                ));
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "symlink rejected inside trusted root: {}",
                    path.display()
                ));
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| format!("non-UTF-8 path rejected inside {}", root.path.display()))?;
            if metadata.is_dir() {
                if name == "__pycache__" || name == "tests" || name.starts_with('.') {
                    continue;
                }
                let canonical = path
                    .canonicalize()
                    .map_err(|error| format!("cannot canonicalize {}: {error}", path.display()))?;
                if !canonical.starts_with(&root.path) {
                    return Err(format!(
                        "directory escaped trusted root: {}",
                        path.display()
                    ));
                }
                pending.push(canonical);
                continue;
            }
            if !metadata.is_file()
                || !name.ends_with(".py")
                || (name != "__init__.py" && name.starts_with('_'))
                || name.ends_with("_test.py")
            {
                continue;
            }
            let canonical = path
                .canonicalize()
                .map_err(|error| format!("cannot canonicalize {}: {error}", path.display()))?;
            if !canonical.starts_with(&root.path) {
                return Err(format!("file escaped trusted root: {}", path.display()));
            }
            files.push(canonical);
        }
    }
    files.sort();
    Ok(files)
}

fn module_path_for(root: &TrustedSkillRoot, path: &Path) -> Result<String, String> {
    let relative = path
        .strip_prefix(&root.path)
        .map_err(|_| format!("file escaped trusted root: {}", path.display()))?;
    let mut parts = relative
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let filename = parts
        .pop()
        .ok_or_else(|| format!("invalid Python source path: {}", path.display()))?;
    if filename != "__init__.py" {
        let stem = filename
            .strip_suffix(".py")
            .ok_or_else(|| format!("invalid Python source path: {}", path.display()))?;
        parts.push(stem.to_string());
    }
    if parts.is_empty() {
        Ok(root.package.clone())
    } else {
        Ok(format!("{}.{}", root.package, parts.join(".")))
    }
}

fn display_source_path(root: &TrustedSkillRoot, path: &Path) -> String {
    let repository_root = if root.package.starts_with("core.") {
        root.path.parent().and_then(Path::parent)
    } else {
        root.path.parent()
    };
    let Some(repository_root) = repository_root else {
        return format!(
            "{}:{}",
            root.kind,
            path.strip_prefix(&root.path).unwrap_or(path).display()
        );
    };
    path.strip_prefix(repository_root)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| {
            format!(
                "{}:{}",
                root.kind,
                path.strip_prefix(&root.path).unwrap_or(path).display()
            )
        })
}

fn read_modules(roots: &[TrustedSkillRoot]) -> Result<Vec<ModuleRecord>, String> {
    let mut modules = Vec::new();
    for root in roots {
        for path in collect_python_files(root)? {
            let metadata = fs::metadata(&path)
                .map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
            if metadata.len() > MAX_SKILL_SOURCE_BYTES {
                return Err(format!(
                    "source exceeds {MAX_SKILL_SOURCE_BYTES} bytes: {}",
                    path.display()
                ));
            }
            let raw = fs::read(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let source = String::from_utf8(raw.clone())
                .map_err(|error| format!("source is not UTF-8 ({}): {error}", path.display()))?;
            let source_path = display_source_path(root, &path);
            let tree = ast::Suite::parse(&source, &source_path)
                .map_err(|error| format!("source parse failed ({}): {error}", path.display()))?;
            modules.push(ModuleRecord {
                path: path.clone(),
                module_path: module_path_for(root, &path)?,
                source_kind: root.kind.clone(),
                source_path,
                source_sha256: format!("{:x}", Sha256::digest(&raw)),
                source,
                tree,
                aliases: BTreeMap::new(),
                local_classes: BTreeSet::new(),
            });
        }
    }
    modules.sort_by(|left, right| {
        (&left.module_path, &left.source_path).cmp(&(&right.module_path, &right.source_path))
    });
    Ok(modules)
}

fn attribute_name(node: &Expr) -> String {
    match node {
        Expr::Name(item) => item.id.to_string(),
        Expr::Attribute(item) => {
            let prefix = attribute_name(&item.value);
            if prefix.is_empty() {
                item.attr.to_string()
            } else {
                format!("{prefix}.{}", item.attr)
            }
        }
        _ => String::new(),
    }
}

fn resolve_symbol(raw: &str, module: &ModuleRecord) -> String {
    let (first, remainder) = raw.split_once('.').unwrap_or((raw, ""));
    if let Some(target) = module.aliases.get(first) {
        if remainder.is_empty() {
            target.clone()
        } else {
            format!("{target}.{remainder}")
        }
    } else if remainder.is_empty() && module.local_classes.contains(first) {
        format!("{}.{}", module.module_path, first)
    } else {
        raw.to_string()
    }
}

fn resolve_from_import(module: &ModuleRecord, imported: &str, level: usize) -> String {
    if level == 0 {
        return imported.to_string();
    }
    let package = if module.path.file_name().and_then(|name| name.to_str()) == Some("__init__.py") {
        module.module_path.as_str()
    } else {
        module
            .module_path
            .rsplit_once('.')
            .map_or("", |(head, _)| head)
    };
    let mut parts = package
        .split('.')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    for _ in 1..level {
        parts.pop();
    }
    if !imported.is_empty() {
        parts.extend(imported.split('.'));
    }
    parts.join(".")
}

fn collect_top_level_imports<'a>(statements: &'a [Stmt], output: &mut Vec<&'a Stmt>) {
    for statement in statements {
        match statement {
            Stmt::Import(_) | Stmt::ImportFrom(_) => output.push(statement),
            Stmt::If(item) => {
                collect_top_level_imports(&item.body, output);
                collect_top_level_imports(&item.orelse, output);
            }
            Stmt::Try(item) => {
                collect_top_level_imports(&item.body, output);
                for handler in &item.handlers {
                    let ExceptHandler::ExceptHandler(handler) = handler;
                    collect_top_level_imports(&handler.body, output);
                }
                collect_top_level_imports(&item.orelse, output);
                collect_top_level_imports(&item.finalbody, output);
            }
            Stmt::TryStar(item) => {
                collect_top_level_imports(&item.body, output);
                for handler in &item.handlers {
                    let ExceptHandler::ExceptHandler(handler) = handler;
                    collect_top_level_imports(&handler.body, output);
                }
                collect_top_level_imports(&item.orelse, output);
                collect_top_level_imports(&item.finalbody, output);
            }
            _ => {}
        }
    }
}

fn literal_metadata(node: &Expr) -> Option<Value> {
    match node {
        Expr::Constant(item) => match &item.value {
            Constant::None => Some(Value::Null),
            Constant::Bool(value) => Some(Value::Bool(*value)),
            Constant::Str(value) => Some(Value::String(value.clone())),
            Constant::Int(value) => {
                let rendered = value.to_string();
                rendered
                    .parse::<i64>()
                    .map(|number| json!(number))
                    .or_else(|_| rendered.parse::<u64>().map(|number| json!(number)))
                    .ok()
            }
            Constant::Float(value) => serde_json::Number::from_f64(*value).map(Value::Number),
            Constant::Tuple(values) => values
                .iter()
                .map(|value| match value {
                    Constant::None => Some(Value::Null),
                    Constant::Bool(item) => Some(Value::Bool(*item)),
                    Constant::Str(item) => Some(Value::String(item.clone())),
                    Constant::Int(item) => item.to_string().parse::<i64>().ok().map(|v| json!(v)),
                    Constant::Float(item) => serde_json::Number::from_f64(*item).map(Value::Number),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .map(Value::Array),
            Constant::Bytes(_) | Constant::Complex { .. } | Constant::Ellipsis => None,
        },
        Expr::List(item) => item
            .elts
            .iter()
            .map(literal_metadata)
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        Expr::Tuple(item) => item
            .elts
            .iter()
            .map(literal_metadata)
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        Expr::Dict(item) => {
            let mut values = serde_json::Map::new();
            for (key, value) in item.keys.iter().zip(&item.values) {
                let key = key
                    .as_ref()
                    .and_then(literal_metadata)?
                    .as_str()?
                    .to_string();
                values.insert(key, literal_metadata(value)?);
            }
            Some(Value::Object(values))
        }
        Expr::UnaryOp(item) => {
            use rustpython_parser::ast::UnaryOp;
            if item.op != UnaryOp::USub {
                return None;
            }
            match literal_metadata(&item.operand)? {
                Value::Number(number) => {
                    if let Some(value) = number.as_i64() {
                        Some(json!(-value))
                    } else {
                        number.as_f64().and_then(|value| {
                            serde_json::Number::from_f64(-value).map(Value::Number)
                        })
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn decorator_metadata(node: &Expr) -> (bool, BTreeMap<String, Value>, Vec<String>) {
    let (target, args, keywords) = match node {
        Expr::Call(item) => (
            item.func.as_ref(),
            item.args.as_slice(),
            item.keywords.as_slice(),
        ),
        _ => (node, &[][..], &[][..]),
    };
    let decorator = attribute_name(target);
    let decorator = decorator.rsplit('.').next().unwrap_or("");
    if !SKILL_DECORATORS.contains(&decorator) {
        return (false, BTreeMap::new(), Vec::new());
    }
    let mut metadata = BTreeMap::new();
    let mut errors = Vec::new();
    if let Some(first) = args.first() {
        match literal_metadata(first) {
            Some(Value::String(value)) => {
                metadata.insert("name".to_string(), Value::String(value));
            }
            _ => errors.push("decorator positional name must be a string literal".to_string()),
        }
    }
    for keyword in keywords {
        let Some(name) = keyword.arg.as_ref().map(ToString::to_string) else {
            continue;
        };
        if !METADATA_FIELDS.contains(&name.as_str()) {
            continue;
        }
        match literal_metadata(&keyword.value) {
            Some(value) => {
                metadata.insert(name, value);
            }
            None => errors.push(format!("decorator metadata {name:?} must be literal")),
        }
    }
    (true, metadata, errors)
}

fn class_metadata(node: &ast::StmtClassDef) -> (BTreeMap<String, Value>, Vec<String>, bool) {
    let mut metadata = BTreeMap::new();
    let mut errors = Vec::new();
    let mut decorated = false;
    for decorator in &node.decorator_list {
        let (recognized, values, decorator_errors) = decorator_metadata(decorator);
        decorated |= recognized;
        metadata.extend(values);
        errors.extend(decorator_errors);
    }
    for statement in &node.body {
        let (targets, value): (Vec<&Expr>, Option<&Expr>) = match statement {
            Stmt::Assign(item) => (item.targets.iter().collect(), Some(item.value.as_ref())),
            Stmt::AnnAssign(item) => (vec![item.target.as_ref()], item.value.as_deref()),
            _ => continue,
        };
        for target in targets {
            let Expr::Name(target) = target else {
                continue;
            };
            let name = target.id.to_string();
            if !METADATA_FIELDS.contains(&name.as_str()) {
                continue;
            }
            match value.and_then(literal_metadata) {
                Some(value) => {
                    metadata.insert(name, value);
                }
                None => errors.push(format!("class metadata {name:?} must be literal")),
            }
        }
    }
    (metadata, errors, decorated)
}

fn meaningful_body(node: &ast::StmtClassDef) -> bool {
    node.body.iter().any(|statement| match statement {
        Stmt::Pass(_) => false,
        Stmt::Expr(item) => !matches!(&*item.value, Expr::Constant(value) if matches!(value.value, Constant::Str(_))),
        Stmt::Assign(item) => !item.targets.iter().all(|target| {
            matches!(target, Expr::Name(name) if METADATA_FIELDS.contains(&name.id.as_str()))
        }),
        Stmt::AnnAssign(item) => {
            !matches!(&*item.target, Expr::Name(name) if METADATA_FIELDS.contains(&name.id.as_str()))
        }
        _ => true,
    })
}

fn line_number(source: &str, node: &impl Ranged) -> usize {
    let offset = u32::from(node.start()) as usize;
    source.as_bytes()[..offset.min(source.len())]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1
}

fn parse_classes(modules: &mut [ModuleRecord]) -> Result<BTreeMap<String, ParsedClass>, String> {
    let mut parsed = BTreeMap::new();
    for module in modules {
        module.local_classes = module
            .tree
            .iter()
            .filter_map(|statement| match statement {
                Stmt::ClassDef(item) => Some(item.name.to_string()),
                _ => None,
            })
            .collect();
        let mut imports = Vec::new();
        collect_top_level_imports(&module.tree, &mut imports);
        for statement in imports {
            match statement {
                Stmt::Import(item) => {
                    for alias in &item.names {
                        let imported = alias.name.to_string();
                        let local = alias
                            .asname
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| {
                                imported.split('.').next().unwrap_or("").to_string()
                            });
                        module.aliases.insert(local, imported);
                    }
                }
                Stmt::ImportFrom(item) => {
                    let imported_module = resolve_from_import(
                        module,
                        item.module.as_ref().map_or("", |value| value.as_str()),
                        item.level.as_ref().map_or(0, |value| value.to_usize()),
                    );
                    for alias in &item.names {
                        if alias.name.as_str() == "*" {
                            continue;
                        }
                        let imported = alias.name.to_string();
                        let local = alias
                            .asname
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| imported.clone());
                        let target = if imported_module.is_empty() {
                            imported
                        } else {
                            format!("{imported_module}.{imported}")
                        };
                        module.aliases.insert(local, target);
                    }
                }
                _ => {}
            }
        }

        for statement in &module.tree {
            let Stmt::ClassDef(node) = statement else {
                continue;
            };
            let (metadata, metadata_errors, decorated) = class_metadata(node);
            let record = ParsedClass {
                module_path: module.module_path.clone(),
                class_name: node.name.to_string(),
                source_kind: module.source_kind.clone(),
                source_path: module.source_path.clone(),
                source_sha256: module.source_sha256.clone(),
                line: line_number(&module.source, node),
                bases: node
                    .bases
                    .iter()
                    .map(attribute_name)
                    .filter(|name| !name.is_empty())
                    .map(|name| resolve_symbol(&name, module))
                    .collect(),
                metadata,
                metadata_errors,
                decorated,
                meaningful_body: meaningful_body(node),
            };
            let qualified = record.qualified_name();
            if parsed.insert(qualified.clone(), record).is_some() {
                return Err(format!(
                    "duplicate qualified class across trusted roots: {qualified}"
                ));
            }
        }
    }
    Ok(parsed)
}

fn is_skill_class(
    qualified: &str,
    classes: &BTreeMap<String, ParsedClass>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    if BASE_SKILL_NAMES.contains(&qualified) {
        return true;
    }
    let Some(record) = classes.get(qualified) else {
        return false;
    };
    if record.decorated {
        return true;
    }
    if !visiting.insert(qualified.to_string()) {
        return false;
    }
    let result = record.bases.iter().any(|base| {
        BASE_SKILL_NAMES.contains(&base.as_str()) || is_skill_class(base, classes, visiting)
    });
    visiting.remove(qualified);
    result
}

fn effective_metadata(
    qualified: &str,
    field: &str,
    classes: &BTreeMap<String, ParsedClass>,
    visiting: &mut BTreeSet<String>,
) -> (Option<Value>, bool) {
    let Some(record) = classes.get(qualified) else {
        return (None, false);
    };
    if let Some(value) = record.metadata.get(field) {
        return (Some(value.clone()), false);
    }
    if !visiting.insert(qualified.to_string()) {
        return (None, false);
    }
    for base in &record.bases {
        if BASE_SKILL_NAMES.contains(&base.as_str()) {
            continue;
        }
        let (value, _) = effective_metadata(base, field, classes, visiting);
        if value.is_some() {
            visiting.remove(qualified);
            return (value, true);
        }
    }
    visiting.remove(qualified);
    (None, false)
}

fn authority_class_for(scope: &str) -> Option<&'static str> {
    match scope {
        "status" | "read_only" | "pure_compute" => Some("observe"),
        "sandboxed_compute" => Some("bounded_compute"),
        "state_mutation" => Some("state_write"),
        "external_io" => Some("external_effect"),
        "read_write_artifacts" => Some("artifact_write"),
        "foreground_desktop_control" | "foreground_browser_dialogue" => Some("foreground_control"),
        "privileged_mutation" => Some("privileged"),
        _ => None,
    }
}

fn legacy_effect_scope(name: &str) -> Option<&'static str> {
    match name {
        "clock" => Some("status"),
        "environment_info"
        | "evolution_status"
        | "free_search"
        | "grounded_search"
        | "local_reference_search"
        | "malware_analysis"
        | "query_beliefs"
        | "query_visual_context"
        | "search_web"
        | "sec_ops"
        | "sovereign_vision"
        | "stealth_ops"
        | "system_proprioception"
        | "web_search" => Some("read_only"),
        "coding_skill" | "native_chat" | "propagation" | "render_bridge" => Some("pure_compute"),
        "branching_futures" | "code_repl" | "internal_sandbox" | "run_code" => {
            Some("sandboxed_compute")
        }
        "ManageAbilities"
        | "add_belief"
        | "cognitive_trainer"
        | "curiosity"
        | "dream_sleep"
        | "execute_nethack_action"
        | "file_operation"
        | "force_dream_cycle"
        | "knowledge_base"
        | "memory_ops"
        | "memory_sync"
        | "personality"
        | "plan_mode"
        | "toggle_senses"
        | "uplink_local" => Some("state_mutation"),
        "browser_action"
        | "delegate_shard"
        | "deploy_ghost_probe"
        | "email_adapter"
        | "embodiment"
        | "inter_agent_comm"
        | "mcp_client"
        | "messages"
        | "network_discovery"
        | "network_recon"
        | "notify_user"
        | "reddit_adapter"
        | "social_lurker"
        | "sovereign_browser"
        | "sovereign_network"
        | "spawn_agent"
        | "spawn_agents_parallel"
        | "speak"
        | "voice_output"
        | "x_tools" => Some("external_io"),
        "build_app"
        | "image_gen"
        | "listen"
        | "manifest_to_device"
        | "manim_renderer"
        | "program_dna_equivalence_battery"
        | "program_dna_reconstruct"
        | "sovereign_imagination"
        | "test_generator" => Some("read_write_artifacts"),
        "computer_use" | "desktop_task" | "os_automation" | "os_manipulation" | "system_ops" => {
            Some("foreground_desktop_control")
        }
        "web_interlocutor" => Some("foreground_browser_dialogue"),
        "auto_refactor" | "improve_own_code" | "install_package" | "network_ops"
        | "self_evolution" | "self_improvement" | "self_repair" | "shell"
        | "sovereign_terminal" | "train_self" => Some("privileged_mutation"),
        _ => None,
    }
}

fn class_exclusion_reason(module: &str, class: &str) -> Option<&'static str> {
    match (module, class) {
        ("skills.browser_action", "UnifiedBrowserSkill") => Some("superseded_by:sovereign_browser"),
        ("skills.network_discovery", "NetworkDiscovery") => Some("superseded_by:sovereign_network"),
        ("skills.network_ops", "NetworkOpsSkill") => Some("superseded_by:sovereign_network"),
        ("skills.network_recon", "NetworkReconSkill") => Some("superseded_by:sovereign_network"),
        ("skills.shell", "ShellSkill") => Some("superseded_by:sovereign_terminal"),
        ("skills.system_ops", "SystemOpsSkill") => Some("superseded_by:computer_use"),
        ("skills.train_self", "TrainSelfSkill") => Some("superseded_by:train_self"),
        _ => None,
    }
}

fn internal_only(name: &str) -> bool {
    matches!(name, "branching_futures" | "manim_renderer" | "mcp_client")
}

fn valid_skill_name(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic())
        && value.len() <= 64
        && chars.all(|item| item == '_' || item == '-' || item.is_ascii_alphanumeric())
}

fn constructor_dependency(value: &Value) -> Option<String> {
    let rendered = match value {
        Value::String(item) => item.trim().to_string(),
        Value::Number(item) => item.to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        Value::Array(_) | Value::Object(_) => return None,
    };
    let has_alphanumeric = rendered.chars().any(char::is_alphanumeric);
    if rendered.is_empty()
        || !has_alphanumeric
        || !rendered
            .chars()
            .all(|item| item == '_' || item.is_alphanumeric())
    {
        None
    } else {
        Some(rendered)
    }
}

fn declarations_from_classes(
    classes: &BTreeMap<String, ParsedClass>,
) -> (Vec<SkillCandidate>, Vec<SkillCandidate>, Vec<CatalogIssue>) {
    let mut eligible = Vec::new();
    let mut excluded = Vec::new();
    let mut issues = Vec::new();
    for (qualified, record) in classes {
        if BASE_SKILL_NAMES.contains(&qualified.as_str())
            || !is_skill_class(qualified, classes, &mut BTreeSet::new())
            || record.class_name.starts_with('_')
            || record.metadata.get("abstract") == Some(&Value::Bool(true))
        {
            continue;
        }
        if !record.metadata_errors.is_empty() {
            issues.push(CatalogIssue::error(
                "dynamic_metadata",
                record.metadata_errors.join("; "),
                Some(record),
            ));
            continue;
        }
        let (name_value, inherited_name) =
            effective_metadata(qualified, "name", classes, &mut BTreeSet::new());
        let Some(Value::String(name)) = name_value else {
            if record.meaningful_body {
                issues.push(CatalogIssue::error(
                    "missing_static_name",
                    "concrete skill classes require a literal name or skill decorator",
                    Some(record),
                ));
            }
            continue;
        };
        if !valid_skill_name(&name) {
            issues.push(CatalogIssue::error(
                "invalid_skill_name",
                format!("invalid literal skill name: {name:?}"),
                Some(record),
            ));
            continue;
        }
        let (description_value, inherited_description) =
            effective_metadata(qualified, "description", classes, &mut BTreeSet::new());
        let Some(Value::String(description)) = description_value else {
            issues.push(CatalogIssue::error(
                "missing_static_description",
                format!("skill {name:?} requires a non-empty literal description"),
                Some(record),
            ));
            continue;
        };
        let description = description.trim().to_string();
        if description.is_empty() {
            issues.push(CatalogIssue::error(
                "missing_static_description",
                format!("skill {name:?} requires a non-empty literal description"),
                Some(record),
            ));
            continue;
        }
        let (scope_value, inherited_scope) =
            effective_metadata(qualified, "effect_scope", classes, &mut BTreeSet::new());
        let declared_scope = scope_value
            .as_ref()
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let effect_scope = if declared_scope.is_empty() {
            legacy_effect_scope(&name).unwrap_or("").to_string()
        } else {
            declared_scope
        };
        let Some(authority_class) = authority_class_for(&effect_scope) else {
            issues.push(CatalogIssue::error(
                "unclassified_effect",
                format!("skill {name:?} needs a recognized literal or catalog effect_scope"),
                Some(record),
            ));
            continue;
        };
        let (dependencies_value, inherited_dependencies) = effective_metadata(
            qualified,
            "constructor_dependencies",
            classes,
            &mut BTreeSet::new(),
        );
        let mut dependencies = Vec::new();
        if let Some(value) = dependencies_value {
            let Some(values) = value.as_array() else {
                issues.push(CatalogIssue::error(
                    "invalid_constructor_dependencies",
                    "constructor_dependencies must be a unique literal list of identifiers",
                    Some(record),
                ));
                continue;
            };
            let mut seen = BTreeSet::new();
            let mut valid = true;
            for value in values {
                let Some(item) = constructor_dependency(value) else {
                    valid = false;
                    break;
                };
                if !seen.insert(item.clone()) {
                    valid = false;
                    break;
                }
                dependencies.push(item);
            }
            if !valid {
                issues.push(CatalogIssue::error(
                    "invalid_constructor_dependencies",
                    "constructor_dependencies must be a unique literal list of identifiers",
                    Some(record),
                ));
                continue;
            }
        }
        let compatibility_wrapper = inherited_name && !record.meaningful_body;
        let exclusion_reason = if compatibility_wrapper {
            "compatibility_wrapper"
        } else if let Some(reason) = class_exclusion_reason(&record.module_path, &record.class_name)
        {
            reason
        } else if internal_only(&name) {
            "internal_only"
        } else {
            ""
        };
        let identity = format!(
            "{}:{}:{}:{}",
            record.module_path, record.class_name, name, record.line
        );
        let candidate = SkillCandidate {
            name,
            description,
            module_path: record.module_path.clone(),
            class_name: record.class_name.clone(),
            source_kind: record.source_kind.clone(),
            source_path: record.source_path.clone(),
            source_sha256: record.source_sha256.clone(),
            line: record.line,
            effect_scope,
            authority_class: authority_class.to_string(),
            constructor_dependencies: dependencies,
            decorated: record.decorated,
            inherited_metadata: inherited_name
                || inherited_description
                || inherited_scope
                || inherited_dependencies,
            exclusion_reason: exclusion_reason.to_string(),
            catalog_id: format!("{:x}", Sha256::digest(identity.as_bytes()))[..20].to_string(),
        };
        if exclusion_reason.is_empty() {
            eligible.push(candidate);
        } else {
            excluded.push(candidate);
        }
    }
    eligible.sort_by(|left, right| {
        (
            left.name.to_ascii_lowercase(),
            &left.name,
            &left.module_path,
            &left.class_name,
            &left.source_path,
            left.line,
        )
            .cmp(&(
                right.name.to_ascii_lowercase(),
                &right.name,
                &right.module_path,
                &right.class_name,
                &right.source_path,
                right.line,
            ))
    });
    excluded.sort_by(|left, right| {
        (
            left.name.to_ascii_lowercase(),
            &left.module_path,
            &left.class_name,
        )
            .cmp(&(
                right.name.to_ascii_lowercase(),
                &right.module_path,
                &right.class_name,
            ))
    });
    (eligible, excluded, issues)
}

fn discover_skill_candidates_json(roots_json: &str) -> Result<String, String> {
    let roots = parse_trusted_roots(roots_json)?;
    let mut modules = read_modules(&roots)?;
    let source_file_count = modules.len();
    let classes = parse_classes(&mut modules)?;
    let (candidates, excluded, mut issues) = declarations_from_classes(&classes);
    let candidate_payload = serde_json::to_string(&json!({"candidates": candidates}))
        .map_err(|error| error.to_string())?;
    let canonical = canonicalize_skill_index_json(&candidate_payload)?;
    let canonical: Value = serde_json::from_str(&canonical).map_err(|error| error.to_string())?;
    for duplicate in canonical
        .get("duplicates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name_key = duplicate
            .get("name_key")
            .and_then(Value::as_str)
            .unwrap_or("");
        issues.push(CatalogIssue::error(
            "duplicate_skill_name",
            format!("ambiguous case-insensitive name {name_key:?}"),
            None,
        ));
    }
    serde_json::to_string(&json!({
        "accepted": canonical.get("accepted").cloned().unwrap_or_else(|| json!([])),
        "backend": "rust-filesystem",
        "candidates": serde_json::from_str::<Value>(&candidate_payload)
            .ok()
            .and_then(|value| value.get("candidates").cloned())
            .unwrap_or_else(|| json!([])),
        "duplicates": canonical.get("duplicates").cloned().unwrap_or_else(|| json!([])),
        "excluded": excluded,
        "issues": issues,
        "source_file_count": source_file_count,
    }))
    .map_err(|error| error.to_string())
}

#[pyfunction]
fn discover_skill_candidates(roots_json: String) -> PyResult<String> {
    discover_skill_candidates_json(&roots_json).map_err(PyValueError::new_err)
}

#[pyfunction]
fn pin_to_p_cores() {
    #[cfg(target_os = "macos")]
    unsafe {
        // Elevate current thread to User Initiated QoS (Apple Silicon P-Cores)
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

#[pyfunction]
fn pin_to_e_cores() {
    #[cfg(target_os = "macos")]
    unsafe {
        // Set current thread to Utility QoS (Apple Silicon E-Cores for low power/IO)
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }
}

#[allow(dead_code)] // used only in the non-aarch64 fallback of neon_dot_product
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Apple Silicon NEON-accelerated dot product (Zero-copy)
#[pyfunction]
fn neon_dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        return neon_dot_product_aarch64(&a, &b);
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        scalar_dot_product(&a, &b)
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn neon_dot_product_aarch64(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0;

    // Process in blocks of 4 using NEON intrinsics
    while i + 4 <= len {
        unsafe {
            let va = vld1q_f32(a[i..].as_ptr());
            let vb = vld1q_f32(b[i..].as_ptr());
            let prod = vmulq_f32(va, vb);
            sum += vaddvq_f32(prod); // Vector across-lane sum
        }
        i += 4;
    }

    // Scalar tail for remaining elements
    for j in i..len {
        sum += a[j] * b[j];
    }
    sum
}

// Fused Euler integration step for the continuous-time unified field.
// Mirrors core/consciousness/unified_field._tick exactly:
//   next[i] = clamp(f[i] + (-decay*f[i] + activity[i] + noise[i]) * dt, -1, 1)
// Zero-copy: reads the numpy buffers directly (PyReadonlyArray1) and returns a
// numpy array, avoiding per-call list marshalling. One tight loop replaces ~4
// numpy temporaries (recurrent+input already folded into `activity`).
#[pyfunction]
fn field_integrate<'py>(
    py: Python<'py>,
    f: PyReadonlyArray1<'py, f32>,
    activity: PyReadonlyArray1<'py, f32>,
    noise: PyReadonlyArray1<'py, f32>,
    decay: f32,
    dt: f32,
) -> Bound<'py, PyArray1<f32>> {
    let f = f.as_slice().unwrap_or(&[]);
    let a = activity.as_slice().unwrap_or(&[]);
    let nz = noise.as_slice().unwrap_or(&[]);
    let n = f.len();
    let mut out = vec![0f32; n];
    for i in 0..n {
        let av = if i < a.len() { a[i] } else { 0.0 };
        let nv = if i < nz.len() { nz[i] } else { 0.0 };
        let df = (-decay * f[i] + av + nv) * dt;
        let mut v = f[i] + df;
        if v > 1.0 {
            v = 1.0;
        } else if v < -1.0 {
            v = -1.0;
        }
        out[i] = v;
    }
    PyArray1::from_vec_bound(py, out)
}

fn candidate_string(candidate: &Value, key: &str) -> String {
    candidate
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn candidate_line(candidate: &Value) -> i64 {
    candidate
        .get("line")
        .and_then(Value::as_i64)
        .unwrap_or_default()
}

fn compare_candidates(left: &Value, right: &Value) -> Ordering {
    let left_name = candidate_string(left, "name");
    let right_name = candidate_string(right, "name");
    (
        left_name.to_ascii_lowercase(),
        left_name,
        candidate_string(left, "module_path"),
        candidate_string(left, "class_name"),
        candidate_string(left, "source_path"),
        candidate_line(left),
    )
        .cmp(&(
            right_name.to_ascii_lowercase(),
            right_name,
            candidate_string(right, "module_path"),
            candidate_string(right, "class_name"),
            candidate_string(right, "source_path"),
            candidate_line(right),
        ))
}

fn canonicalize_skill_index_json(candidate_json: &str) -> Result<String, String> {
    let payload: Value = serde_json::from_str(candidate_json).map_err(|error| error.to_string())?;
    let mut candidates = payload
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| "skill index payload must contain a candidates array".to_string())?;
    candidates.sort_by(compare_candidates);

    let mut grouped: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for candidate in candidates {
        let name_key = candidate_string(&candidate, "name").to_ascii_lowercase();
        grouped.entry(name_key).or_default().push(candidate);
    }

    let mut accepted = Vec::new();
    let mut duplicates = Vec::new();
    for (name_key, mut group) in grouped {
        if group.len() == 1 {
            accepted.push(group.remove(0));
        } else {
            duplicates.push(json!({"candidates": group, "name_key": name_key}));
        }
    }
    serde_json::to_string(&json!({"accepted": accepted, "duplicates": duplicates}))
        .map_err(|error| error.to_string())
}

#[pyfunction]
#[pyo3(signature = (catalog_json=None))]
fn build_skill_index(py: Python<'_>, catalog_json: Option<String>) -> PyResult<PyObject> {
    let explicit_payload = catalog_json.is_some();
    let discovery = py.import_bound("core.skills.discovery")?;
    let candidate_json = match catalog_json {
        Some(payload) => payload,
        None => discovery
            .getattr("skill_index_candidates_json")?
            .call0()?
            .extract::<String>()?,
    };
    let canonical =
        canonicalize_skill_index_json(&candidate_json).map_err(PyValueError::new_err)?;
    if explicit_payload {
        return Ok(canonical.into_py(py));
    }
    let index = discovery
        .getattr("_index_dict_from_canonical_json")?
        .call1((canonical,))?;
    Ok(index.into_py(py))
}

#[pymodule]
fn aura_m1_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pin_to_p_cores, m)?)?;
    m.add_function(wrap_pyfunction!(pin_to_e_cores, m)?)?;
    m.add_function(wrap_pyfunction!(neon_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(field_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(build_skill_index, m)?)?;
    m.add_function(wrap_pyfunction!(discover_skill_candidates, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{canonicalize_skill_index_json, discover_skill_candidates_json};
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_ROOT: AtomicU64 = AtomicU64::new(0);

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            let ordinal = NEXT_TEST_ROOT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
                "aura-rust-skill-discovery-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            Self(root)
        }

        fn write(&self, relative: &str, source: &str) {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, source).unwrap();
        }

        fn roots_json(&self, package: &str) -> String {
            serde_json::to_string(&serde_json::json!([{
                "path": self.0.to_str().unwrap(),
                "package": package,
                "kind": "test"
            }]))
            .unwrap()
        }

        fn discover(&self, package: &str) -> Value {
            serde_json::from_str(
                &discover_skill_candidates_json(&self.roots_json(package)).unwrap(),
            )
            .unwrap()
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn names(payload: &Value, field: &str) -> Vec<String> {
        payload[field]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["name"].as_str().unwrap().to_string())
            .collect()
    }

    fn canonical_json(value: &Value) -> String {
        match value {
            Value::Array(items) => format!(
                "[{}]",
                items
                    .iter()
                    .map(canonical_json)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Value::Object(items) => {
                let sorted = items.iter().collect::<std::collections::BTreeMap<_, _>>();
                format!(
                    "{{{}}}",
                    sorted
                        .into_iter()
                        .map(|(key, value)| format!(
                            "{}:{}",
                            serde_json::to_string(key).unwrap(),
                            canonical_json(value)
                        ))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            _ => serde_json::to_string(value).unwrap(),
        }
    }

    #[test]
    fn canonicalizer_sorts_and_rejects_case_insensitive_duplicates() {
        let input = r#"{"candidates":[
            {"name":"zeta","module_path":"skills.z","class_name":"Z","source_path":"z.py","line":2},
            {"name":"Alpha","module_path":"skills.a","class_name":"A","source_path":"a.py","line":1},
            {"name":"alpha","module_path":"skills.b","class_name":"B","source_path":"b.py","line":1}
        ]}"#;
        let output: Value =
            serde_json::from_str(&canonicalize_skill_index_json(input).unwrap()).unwrap();
        assert_eq!(output["accepted"].as_array().unwrap().len(), 1);
        assert_eq!(output["accepted"][0]["name"], "zeta");
        assert_eq!(output["duplicates"].as_array().unwrap().len(), 1);
        assert_eq!(output["duplicates"][0]["name_key"], "alpha");
        assert_eq!(output["duplicates"][0]["candidates"][0]["name"], "Alpha");
    }

    #[test]
    fn discovers_module_level_inherited_and_decorated_skills() {
        let root = TestRoot::new();
        root.write(
            "skills.py",
            r#"
from core.skills.base_skill import BaseSkill as RootSkill

class Template(RootSkill):
    abstract = True
    name = "inherited"
    description = "Inherited metadata"
    effect_scope = "read_only"
    constructor_dependencies = ["memory_service"]

class Inherited(Template):
    def execute(self):
        return None

@aura_skill(name="decorated", description="Decorator metadata", effect_scope="status")
class Decorated:
    pass
"#,
        );

        let payload = root.discover("fixture");
        assert_eq!(names(&payload, "accepted"), vec!["decorated", "inherited"]);
        let inherited = payload["accepted"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["name"] == "inherited")
            .unwrap();
        assert_eq!(
            inherited["constructor_dependencies"],
            serde_json::json!(["memory_service"])
        );
        assert_eq!(inherited["inherited_metadata"], true);
        assert_eq!(payload["issues"], serde_json::json!([]));
    }

    #[test]
    fn supports_assign_annassign_and_multiple_classes() {
        let root = TestRoot::new();
        root.write(
            "many.py",
            r#"
from core.skills.base_skill import BaseSkill

class Alpha(BaseSkill):
    name: str = "alpha"
    description = "Alpha description"
    effect_scope: str = "pure_compute"
    def execute(self): pass

class Beta(BaseSkill):
    name = "beta"
    description: str = "Beta description"
    effect_scope = "state_mutation"
    constructor_dependencies = ("state_store", 7, True)
    def execute(self): pass
"#,
        );

        let payload = root.discover("fixture");
        assert_eq!(names(&payload, "accepted"), vec!["alpha", "beta"]);
        assert_eq!(payload["accepted"][1]["authority_class"], "state_write");
        assert_eq!(
            payload["accepted"][1]["constructor_dependencies"],
            serde_json::json!(["state_store", "7", "True"])
        );
    }

    #[test]
    fn excludes_nested_classes_and_ignored_paths() {
        let root = TestRoot::new();
        root.write(
            "visible.py",
            r#"
from core.skills.base_skill import BaseSkill
class Visible(BaseSkill):
    name = "visible"
    description = "Visible"
    effect_scope = "status"
    def execute(self): pass

def factory():
    class Nested(BaseSkill):
        name = "nested"
        description = "Nested"
        effect_scope = "status"
    return Nested
"#,
        );
        root.write(
            "tests/hidden.py",
            "@skill(name='hidden', description='Hidden', effect_scope='status')\nclass Hidden: pass\n",
        );
        root.write(
            "_private.py",
            "@skill(name='private', description='Private', effect_scope='status')\nclass Private: pass\n",
        );
        root.write(
            "legacy_test.py",
            "@skill(name='legacy', description='Legacy', effect_scope='status')\nclass Legacy: pass\n",
        );

        let payload = root.discover("fixture");
        assert_eq!(names(&payload, "accepted"), vec!["visible"]);
        assert_eq!(payload["source_file_count"], 1);
    }

    #[test]
    fn maps_package_init_modules_and_relative_inheritance() {
        let root = TestRoot::new();
        root.write(
            "base.py",
            r#"
from core.skills.base_skill import BaseSkill
class PackageBase(BaseSkill):
    abstract = True
    name = "package_skill"
    description = "Package skill"
    effect_scope = "read_only"
"#,
        );
        root.write(
            "nested/__init__.py",
            r#"
from ..base import PackageBase
class PackageSkill(PackageBase):
    def execute(self): pass
"#,
        );

        let payload = root.discover("fixture.skills");
        assert_eq!(names(&payload, "accepted"), vec!["package_skill"]);
        assert_eq!(
            payload["accepted"][0]["module_path"],
            "fixture.skills.nested"
        );
        let expected_source = format!(
            "{}/nested/__init__.py",
            root.0.file_name().unwrap().to_string_lossy()
        );
        assert_eq!(payload["accepted"][0]["source_path"], expected_source);
    }

    #[test]
    fn reports_case_insensitive_duplicates_deterministically() {
        let root = TestRoot::new();
        root.write(
            "a.py",
            "@skill(name='Alpha', description='A', effect_scope='status')\nclass A: pass\n",
        );
        root.write(
            "b.py",
            "@skill(name='alpha', description='B', effect_scope='status')\nclass B: pass\n",
        );
        root.write(
            "z.py",
            "@skill(name='zeta', description='Z', effect_scope='status')\nclass Z: pass\n",
        );

        let payload = root.discover("fixture");
        assert_eq!(names(&payload, "accepted"), vec!["zeta"]);
        assert_eq!(payload["duplicates"].as_array().unwrap().len(), 1);
        assert_eq!(payload["duplicates"][0]["name_key"], "alpha");
        assert_eq!(payload["duplicates"][0]["candidates"][0]["name"], "Alpha");
        assert_eq!(payload["issues"][0]["code"], "duplicate_skill_name");
    }

    #[test]
    fn rejects_parse_errors_instead_of_returning_a_partial_catalog() {
        let root = TestRoot::new();
        root.write(
            "valid.py",
            "@skill(name='valid', description='Valid', effect_scope='status')\nclass Valid: pass\n",
        );
        root.write("broken.py", "class Broken(\n");
        let error = discover_skill_candidates_json(&root.roots_json("fixture")).unwrap_err();
        assert!(error.contains("source parse failed"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_anywhere_below_a_trusted_root() {
        use std::os::unix::fs::symlink;

        let root = TestRoot::new();
        let outside = TestRoot::new();
        outside.write("outside.py", "pass\n");
        symlink(outside.0.join("outside.py"), root.0.join("linked.py")).unwrap();
        let error = discover_skill_candidates_json(&root.roots_json("fixture")).unwrap_err();
        assert!(error.contains("symlink rejected"), "{error}");
    }

    #[test]
    fn rejects_malformed_and_overlapping_root_contracts() {
        let root = TestRoot::new();
        fs::create_dir_all(root.0.join("nested")).unwrap();
        let malformed = serde_json::json!([{
            "path": root.0,
            "package": "bad-package",
            "kind": "test"
        }]);
        assert!(discover_skill_candidates_json(&malformed.to_string()).is_err());

        let overlap = serde_json::json!([
            {"path": root.0, "package": "fixture", "kind": "test"},
            {"path": root.0.join("nested"), "package": "fixture.nested", "kind": "test"}
        ]);
        let error = discover_skill_candidates_json(&overlap.to_string()).unwrap_err();
        assert!(error.contains("overlap"), "{error}");

        let escaped = serde_json::json!([{
            "path": root.0.join("nested/.."),
            "package": "fixture",
            "kind": "test"
        }]);
        let error = discover_skill_candidates_json(&escaped.to_string()).unwrap_err();
        assert!(error.contains("not lexically canonical"), "{error}");
    }

    #[test]
    fn checked_in_roots_match_the_certified_catalog_shape() {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .unwrap()
            .canonicalize()
            .unwrap();
        let roots = serde_json::json!([
            {
                "path": repository.join("core/skills"),
                "package": "core.skills",
                "kind": "core"
            },
            {
                "path": repository.join("skills"),
                "package": "skills",
                "kind": "project"
            }
        ]);
        let payload: Value =
            serde_json::from_str(&discover_skill_candidates_json(&roots.to_string()).unwrap())
                .unwrap();
        assert_eq!(payload["source_file_count"], 135);
        assert_eq!(payload["candidates"].as_array().unwrap().len(), 79);
        assert_eq!(payload["accepted"].as_array().unwrap().len(), 79);
        assert_eq!(payload["excluded"].as_array().unwrap().len(), 10);
        assert_eq!(payload["duplicates"], serde_json::json!([]));
        assert_eq!(payload["issues"], serde_json::json!([]));
        let canonical_payload = serde_json::json!({
            "accepted": payload["accepted"],
            "candidate_count": payload["candidates"].as_array().unwrap().len()
                + payload["excluded"].as_array().unwrap().len(),
            "duplicate_count": payload["duplicates"].as_array().unwrap().len(),
            "excluded": payload["excluded"],
            "issues": payload["issues"],
            "source_file_count": payload["source_file_count"],
        });
        let digest = format!("{:x}", Sha256::digest(canonical_json(&canonical_payload)));
        assert_eq!(
            digest,
            // Python's identical canonical payload with ensure_ascii=False.
            "7f54a30e9f954c36870fc758b9463db4d2afc5851527948f13e607b8b3a3dfe3"
        );
    }
}
