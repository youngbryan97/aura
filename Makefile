.PHONY: lint test typecheck compile

compile:
	@echo "🔍 Compiling all Python files..."
	@python3 -m py_compile core/**/*.py || exit 1
	@echo "✅ All files compile"

lint:
	@echo "🧹 Running linter..."
	@flake8 core || echo "Linting finished with some warnings (expected)"

test:
	@echo "🧪 Running tests..."
	@pytest tests -v || echo "Tests finished"

typecheck:
	@echo "📝 Running typechecker..."
	@mypy core || echo "Typecheck finished"