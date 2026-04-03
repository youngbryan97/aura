import AppKit
import Darwin
import Foundation

private let pollInterval: TimeInterval = 0.8
private let bootMarkerTTL: TimeInterval = 180.0
private let readyCloseDelay: TimeInterval = 3.0
private let unhealthyBootWindow: TimeInterval = 90.0
private let stalledRecoveryWindow: TimeInterval = 300.0

private struct BootSnapshot {
    let statusCode: Int
    let payload: [String: Any]

    var bootPhase: String {
        String(describing: payload["boot_phase"] ?? "kernel_bootstrap")
    }

    var statusMessage: String {
        if let text = payload["status_message"] as? String, !text.isEmpty {
            return text
        }
        return "Aura is booting…"
    }

    var progress: Double {
        if let value = payload["progress"] as? NSNumber {
            return value.doubleValue
        }
        if let value = payload["progress"] as? Double {
            return value
        }
        if let value = payload["progress"] as? Int {
            return Double(value)
        }
        return 8.0
    }

    var conversationReady: Bool {
        (payload["conversation_ready"] as? Bool) == true
    }

    var systemReady: Bool {
        (payload["system_ready"] as? Bool) == true
    }

    var status: String {
        String(describing: payload["status"] ?? "booting")
    }

    var semver: String {
        String(describing: payload["semver"] ?? "")
    }

    var runtimeAge: TimeInterval {
        if let value = payload["runtime_age_s"] as? NSNumber {
            return value.doubleValue
        }
        if let value = payload["runtime_age_s"] as? Double {
            return value
        }
        if let orchestrator = payload["orchestrator"] as? [String: Any] {
            if let value = orchestrator["uptime"] as? NSNumber {
                return value.doubleValue
            }
            if let value = orchestrator["uptime"] as? Double {
                return value
            }
        }
        return 0.0
    }

    var runtimeIntegrityOK: Bool {
        guard let checks = payload["checks"] as? [String: Any] else {
            return true
        }
        return (checks["runtime_integrity"] as? Bool) ?? true
    }

    var lastFailureReason: String {
        if let lane = payload["conversation_lane"] as? [String: Any],
           let reason = lane["last_failure_reason"] as? String {
            return reason
        }
        return ""
    }

    var launcherReady: Bool {
        if let value = payload["launcher_ready"] as? Bool {
            return value
        }
        let normalized = bootPhase.lowercased()
        if (payload["ready"] as? Bool) == true {
            return true
        }
        return normalized == "kernel_ready" || normalized == "proxy_ready"
    }

    var blockers: [String] {
        (payload["blockers"] as? [String]) ?? []
    }

    var phaseDisplay: String {
        bootPhase.replacingOccurrences(of: "_", with: " ")
    }

    func replacementReason(expectedSemver: String) -> String? {
        let trimmedExpected = expectedSemver.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedExpected = trimmedExpected.split(separator: "-", maxSplits: 1).first.map(String.init) ?? trimmedExpected
        let normalizedServed = semver.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "-", maxSplits: 1).first.map(String.init) ?? semver
        if !normalizedExpected.isEmpty, !normalizedServed.isEmpty, normalizedServed != normalizedExpected {
            return "Existing runtime is serving build \(semver), but launcher expects \(trimmedExpected)."
        }

        let normalized = bootPhase.lowercased()
        if !runtimeIntegrityOK && runtimeAge >= 60.0 {
            return "Existing runtime lost integrity markers and should be refreshed."
        }
        if statusCode >= 500 && runtimeAge >= unhealthyBootWindow {
            return "Existing runtime has been unhealthy for too long."
        }
        if normalized == "conversation_recovering" && !conversationReady && runtimeAge >= stalledRecoveryWindow {
            return "Conversation lane has been recovering for too long."
        }
        if normalized == "kernel_warming" && runtimeAge >= stalledRecoveryWindow {
            return "Kernel boot has been warming too long without becoming healthy."
        }
        return nil
    }
}

private enum LaunchAttemptResult {
    case launched
    case observingExistingBoot
    case failed(String)
}

private extension NSColor {
    static let auraCanvasTop = NSColor(calibratedRed: 0.07, green: 0.08, blue: 0.13, alpha: 1.0)
    static let auraCanvasBottom = NSColor(calibratedRed: 0.03, green: 0.04, blue: 0.08, alpha: 1.0)
    static let auraPanel = NSColor(calibratedRed: 0.09, green: 0.11, blue: 0.16, alpha: 0.86)
    static let auraPanelBorder = NSColor(calibratedRed: 0.38, green: 0.30, blue: 0.76, alpha: 0.36)
    static let auraTrack = NSColor(calibratedRed: 0.16, green: 0.18, blue: 0.25, alpha: 1.0)
    static let auraCyan = NSColor(calibratedRed: 0.18, green: 0.86, blue: 1.0, alpha: 1.0)
    static let auraBlue = NSColor(calibratedRed: 0.18, green: 0.49, blue: 1.0, alpha: 1.0)
    static let auraViolet = NSColor(calibratedRed: 0.61, green: 0.36, blue: 1.0, alpha: 1.0)
    static let auraTextMuted = NSColor(calibratedWhite: 0.74, alpha: 1.0)
}

private final class LauncherBackgroundView: NSView {
    private let gradientLayer = CAGradientLayer()
    private let glowPrimary = CALayer()
    private let glowSecondary = CALayer()
    private let vignetteLayer = CAGradientLayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer = CALayer()
        layer?.masksToBounds = true
        layer?.cornerRadius = 22

        gradientLayer.colors = [NSColor.auraCanvasTop.cgColor, NSColor.auraCanvasBottom.cgColor]
        gradientLayer.startPoint = CGPoint(x: 0.0, y: 1.0)
        gradientLayer.endPoint = CGPoint(x: 1.0, y: 0.0)
        layer?.addSublayer(gradientLayer)

        glowPrimary.backgroundColor = NSColor.auraViolet.withAlphaComponent(0.22).cgColor
        glowPrimary.cornerRadius = 150
        glowPrimary.shadowColor = NSColor.auraViolet.cgColor
        glowPrimary.shadowOpacity = 0.9
        glowPrimary.shadowRadius = 80
        glowPrimary.shadowOffset = .zero
        layer?.addSublayer(glowPrimary)

        glowSecondary.backgroundColor = NSColor.auraCyan.withAlphaComponent(0.16).cgColor
        glowSecondary.cornerRadius = 120
        glowSecondary.shadowColor = NSColor.auraCyan.cgColor
        glowSecondary.shadowOpacity = 0.8
        glowSecondary.shadowRadius = 60
        glowSecondary.shadowOffset = .zero
        layer?.addSublayer(glowSecondary)

        vignetteLayer.colors = [
            NSColor.black.withAlphaComponent(0.0).cgColor,
            NSColor.black.withAlphaComponent(0.28).cgColor,
        ]
        vignetteLayer.startPoint = CGPoint(x: 0.5, y: 1.0)
        vignetteLayer.endPoint = CGPoint(x: 0.5, y: 0.0)
        layer?.addSublayer(vignetteLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        gradientLayer.frame = bounds
        vignetteLayer.frame = bounds
        glowPrimary.frame = CGRect(x: bounds.maxX - 220, y: bounds.maxY - 210, width: 210, height: 210)
        glowSecondary.frame = CGRect(x: 36, y: bounds.maxY - 180, width: 170, height: 170)
    }
}

private final class GradientProgressBar: NSView {
    private let trackLayer = CALayer()
    private let fillGlowLayer = CALayer()
    private let fillGradientLayer = CAGradientLayer()

    var progress: Double = 0 {
        didSet {
            needsLayout = true
        }
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer = CALayer()
        layer?.masksToBounds = false

        trackLayer.backgroundColor = NSColor.auraTrack.cgColor
        trackLayer.cornerRadius = 9
        layer?.addSublayer(trackLayer)

        fillGlowLayer.backgroundColor = NSColor.auraBlue.withAlphaComponent(0.55).cgColor
        fillGlowLayer.cornerRadius = 9
        fillGlowLayer.shadowColor = NSColor.auraViolet.cgColor
        fillGlowLayer.shadowOpacity = 0.9
        fillGlowLayer.shadowRadius = 12
        fillGlowLayer.shadowOffset = .zero
        layer?.addSublayer(fillGlowLayer)

        fillGradientLayer.colors = [
            NSColor.auraCyan.cgColor,
            NSColor.auraBlue.cgColor,
            NSColor.auraViolet.cgColor,
        ]
        fillGradientLayer.startPoint = CGPoint(x: 0.0, y: 0.5)
        fillGradientLayer.endPoint = CGPoint(x: 1.0, y: 0.5)
        fillGradientLayer.cornerRadius = 9
        layer?.addSublayer(fillGradientLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        let rect = bounds.insetBy(dx: 0, dy: 3)
        trackLayer.frame = rect

        let clamped = max(0.0, min(100.0, progress))
        let fillWidth = max(18.0, rect.width * CGFloat(clamped / 100.0))
        let fillRect = CGRect(x: rect.minX, y: rect.minY, width: min(fillWidth, rect.width), height: rect.height)

        fillGlowLayer.isHidden = clamped <= 0.0
        fillGradientLayer.isHidden = clamped <= 0.0
        fillGlowLayer.frame = fillRect
        fillGradientLayer.frame = fillRect
    }
}

private final class CapsuleButton: NSButton {
    enum Style {
        case accent
        case secondary
        case subtle
        case danger
    }

    private let style: Style

    init(title: String, style: Style, target: AnyObject?, action: Selector) {
        self.style = style
        super.init(frame: .zero)
        self.title = title
        self.target = target
        self.action = action
        isBordered = false
        bezelStyle = .regularSquare
        focusRingType = .none
        wantsLayer = true
        layer?.cornerRadius = 14
        layer?.borderWidth = 1
        updateAppearance()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var intrinsicContentSize: NSSize {
        let base = super.intrinsicContentSize
        return NSSize(width: base.width + 30, height: 42)
    }

    private func updateAppearance() {
        let font = NSFont.systemFont(ofSize: 15, weight: .semibold)
        switch style {
        case .accent:
            layer?.backgroundColor = NSColor.auraViolet.withAlphaComponent(0.22).cgColor
            layer?.borderColor = NSColor.auraViolet.withAlphaComponent(0.44).cgColor
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: NSColor.white,
                ]
            )
        case .secondary:
            layer?.backgroundColor = NSColor.white.withAlphaComponent(0.06).cgColor
            layer?.borderColor = NSColor.white.withAlphaComponent(0.10).cgColor
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: NSColor(calibratedWhite: 0.92, alpha: 1.0),
                ]
            )
        case .subtle:
            layer?.backgroundColor = NSColor.black.withAlphaComponent(0.14).cgColor
            layer?.borderColor = NSColor.white.withAlphaComponent(0.08).cgColor
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: NSColor.auraTextMuted,
                ]
            )
        case .danger:
            layer?.backgroundColor = NSColor(calibratedRed: 0.46, green: 0.14, blue: 0.21, alpha: 0.34).cgColor
            layer?.borderColor = NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.67, alpha: 0.36).cgColor
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: NSColor(calibratedRed: 1.0, green: 0.87, blue: 0.91, alpha: 1.0),
                ]
            )
        }
    }
}

final class AuraLauncherDelegate: NSObject, NSApplicationDelegate {
    private enum BadgeStyle {
        case violet
        case cyan
        case blue
        case emerald
        case rose

        var color: NSColor {
            switch self {
            case .violet:
                return .auraViolet
            case .cyan:
                return .auraCyan
            case .blue:
                return .auraBlue
            case .emerald:
                return NSColor(calibratedRed: 0.32, green: 0.93, blue: 0.72, alpha: 1.0)
            case .rose:
                return NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.67, alpha: 1.0)
            }
        }
    }

    private let fileManager = FileManager.default
    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 1.0
        config.timeoutIntervalForResource = 1.5
        return URLSession(configuration: config)
    }()

    private var window: NSWindow!
    private var titleLabel: NSTextField!
    private var detailLabel: NSTextField!
    private var footerLabel: NSTextField!
    private var phaseBadge: NSTextField!
    private var progressIndicator: GradientProgressBar!
    private var progressValueLabel: NSTextField!
    private var openLogsButton: NSButton!
    private var openDesktopButton: NSButton!
    private var openBrowserButton: NSButton!
    private var forceStopButton: NSButton!

    private var auraRoot: URL!
    private var launchScript: URL!
    private var pythonExecutable: URL!
    private var auraMainScript: URL!
    private var logFile: URL!
    private var lockDirectory: URL!
    private var bootMarkerFile: URL!
    private var terminalHandoffMarkerFile: URL!
    private var spawnLockFile: URL!

    private var pollTimer: Timer?
    private var isPolling = false
    private var launchInFlight = false
    private var closeScheduled = false
    private var lastSnapshot: BootSnapshot?
    private var bundledSemver: String = ""
    private var bundledVersionLabel: String = ""
    private var forcedRelaunchAttempted = false
    private var autoDesktopOpenTriggered = false
    private var spawnedFreshRuntime = false
    private let launchedAt = Date()
    private let staleMarkerWithoutRuntimeWindow: TimeInterval = 8.0
    private let terminalHandoffWindow: TimeInterval = 75.0

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            try configurePaths()
        } catch {
            showFatalError(
                title: "Aura Launcher Error",
                detail: error.localizedDescription,
            )
            return
        }

        buildWindow()
        renderTitle("Aura is waking up")
        renderStatus(
            detail: "Preparing the boot monitor…",
            footer: "You can keep this window open while Aura boots.",
            progress: 6.0,
            phase: "launcher online",
            badgeStyle: .violet,
        )
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [])
        NSApp.requestUserAttention(.informationalRequest)

        pollNow()
        pollTimer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] _ in
            self?.pollNow()
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        guard let window else {
            return false
        }
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [])
        NSApp.requestUserAttention(.informationalRequest)
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func configurePaths() throws {
        guard let resourcesURL = Bundle.main.resourceURL else {
            throw NSError(domain: "AuraLauncher", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Bundle resources are missing.",
            ])
        }

        let rootLink = resourcesURL.appendingPathComponent("aura-root")
        let rootFallback = resourcesURL.appendingPathComponent("aura-root-path")

        if let destination = try? fileManager.destinationOfSymbolicLink(atPath: rootLink.path), !destination.isEmpty {
            auraRoot = URL(fileURLWithPath: destination, isDirectory: true)
        } else if let text = try? String(contentsOf: rootFallback, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw NSError(domain: "AuraLauncher", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Aura root path is empty. Rebuild the launcher from the repo.",
                ])
            }
            auraRoot = URL(fileURLWithPath: trimmed, isDirectory: true)
        } else {
            throw NSError(domain: "AuraLauncher", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Aura root path is missing. Rebuild the launcher from the repo.",
            ])
        }

        launchScript = auraRoot.appendingPathComponent("launch_aura.sh")
        auraMainScript = auraRoot.appendingPathComponent("aura_main.py")
        pythonExecutable = try resolvePythonExecutable()

        let semverFile = resourcesURL.appendingPathComponent("aura-version")
        if let text = try? String(contentsOf: semverFile, encoding: .utf8) {
            bundledSemver = text.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            bundledSemver = (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? ""
        }

        let versionLabelFile = resourcesURL.appendingPathComponent("aura-version-full")
        if let text = try? String(contentsOf: versionLabelFile, encoding: .utf8) {
            bundledVersionLabel = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let auraHome = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".aura", isDirectory: true)
        let logDirectory = auraHome.appendingPathComponent("logs", isDirectory: true)
        lockDirectory = auraHome.appendingPathComponent("locks", isDirectory: true)
        logFile = logDirectory.appendingPathComponent("desktop-launch.log")
        bootMarkerFile = lockDirectory.appendingPathComponent("desktop-app-launch.marker")
        terminalHandoffMarkerFile = lockDirectory.appendingPathComponent("desktop-terminal-launch.marker")
        spawnLockFile = lockDirectory.appendingPathComponent("desktop-app-launch.lock")

        try fileManager.createDirectory(at: logDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: lockDirectory, withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: logFile.path) {
            fileManager.createFile(atPath: logFile.path, contents: Data())
        }
    }

    private func buildWindow() {
        let frame = NSRect(x: 0, y: 0, width: 720, height: 412)
        window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Aura"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.backgroundColor = .clear
        window.isOpaque = false
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true

        let contentView = LauncherBackgroundView(frame: frame)
        window.contentView = contentView

        let contentCard = NSView()
        contentCard.translatesAutoresizingMaskIntoConstraints = false
        contentCard.wantsLayer = true
        contentCard.layer?.backgroundColor = NSColor.auraPanel.cgColor
        contentCard.layer?.cornerRadius = 28
        contentCard.layer?.borderColor = NSColor.auraPanelBorder.cgColor
        contentCard.layer?.borderWidth = 1
        contentView.addSubview(contentCard)

        let eyebrowLabel = NSTextField(labelWithString: "AURA LAUNCHER")
        eyebrowLabel.translatesAutoresizingMaskIntoConstraints = false
        eyebrowLabel.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .semibold)
        eyebrowLabel.textColor = NSColor.auraCyan.withAlphaComponent(0.82)

        let iconPlate = NSView()
        iconPlate.translatesAutoresizingMaskIntoConstraints = false
        iconPlate.wantsLayer = true
        iconPlate.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.22).cgColor
        iconPlate.layer?.cornerRadius = 24
        iconPlate.layer?.borderColor = NSColor.auraViolet.withAlphaComponent(0.40).cgColor
        iconPlate.layer?.borderWidth = 1
        contentCard.addSubview(iconPlate)

        let iconView = NSImageView()
        iconView.translatesAutoresizingMaskIntoConstraints = false
        iconView.imageScaling = .scaleProportionallyUpOrDown
        if let iconURL = Bundle.main.url(forResource: "Aura", withExtension: "icns"),
           let icon = NSImage(contentsOf: iconURL) {
            iconView.image = icon
        } else {
            iconView.image = NSApp.applicationIconImage
        }
        iconPlate.addSubview(iconView)

        titleLabel = NSTextField(labelWithString: "Aura is waking up")
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.font = NSFont.systemFont(ofSize: 30, weight: .bold)
        titleLabel.textColor = .white
        titleLabel.maximumNumberOfLines = 2
        titleLabel.lineBreakMode = .byWordWrapping

        detailLabel = NSTextField(wrappingLabelWithString: "Preparing the boot monitor…")
        detailLabel.translatesAutoresizingMaskIntoConstraints = false
        detailLabel.font = NSFont.systemFont(ofSize: 17, weight: .medium)
        detailLabel.textColor = NSColor(calibratedWhite: 0.90, alpha: 1.0)
        detailLabel.maximumNumberOfLines = 4
        detailLabel.lineBreakMode = .byWordWrapping

        phaseBadge = NSTextField(labelWithString: "LAUNCHING")
        phaseBadge.translatesAutoresizingMaskIntoConstraints = false
        phaseBadge.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .semibold)
        phaseBadge.alignment = .center
        phaseBadge.textColor = NSColor.white
        phaseBadge.wantsLayer = true
        phaseBadge.layer?.backgroundColor = NSColor.auraViolet.withAlphaComponent(0.22).cgColor
        phaseBadge.layer?.borderColor = NSColor.auraViolet.withAlphaComponent(0.42).cgColor
        phaseBadge.layer?.borderWidth = 1
        phaseBadge.layer?.cornerRadius = 13

        footerLabel = NSTextField(wrappingLabelWithString: "You can keep this window open while Aura boots.")
        footerLabel.translatesAutoresizingMaskIntoConstraints = false
        footerLabel.font = NSFont.systemFont(ofSize: 14, weight: .regular)
        footerLabel.textColor = NSColor.auraTextMuted
        footerLabel.maximumNumberOfLines = 3
        footerLabel.lineBreakMode = .byWordWrapping

        progressIndicator = GradientProgressBar()
        progressIndicator.translatesAutoresizingMaskIntoConstraints = false
        progressIndicator.progress = 6

        let progressBelowBadge = progressIndicator.topAnchor.constraint(equalTo: phaseBadge.bottomAnchor, constant: 22)
        progressBelowBadge.priority = .defaultHigh
        let progressBelowIcon = progressIndicator.topAnchor.constraint(greaterThanOrEqualTo: iconPlate.bottomAnchor, constant: 22)

        progressValueLabel = NSTextField(labelWithString: "6%")
        progressValueLabel.translatesAutoresizingMaskIntoConstraints = false
        progressValueLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 18, weight: .semibold)
        progressValueLabel.textColor = NSColor.white

        openLogsButton = CapsuleButton(title: "Open Logs", style: .secondary, target: self, action: #selector(openLogs))
        openLogsButton.translatesAutoresizingMaskIntoConstraints = false

        openDesktopButton = CapsuleButton(title: "Open Aura", style: .accent, target: self, action: #selector(openDesktopWindow))
        openDesktopButton.translatesAutoresizingMaskIntoConstraints = false

        openBrowserButton = CapsuleButton(title: "Open Browser", style: .secondary, target: self, action: #selector(openBrowser))
        openBrowserButton.translatesAutoresizingMaskIntoConstraints = false

        forceStopButton = CapsuleButton(title: "Force Stop", style: .danger, target: self, action: #selector(forceStopAura))
        forceStopButton.translatesAutoresizingMaskIntoConstraints = false

        contentCard.addSubview(eyebrowLabel)
        contentCard.addSubview(titleLabel)
        contentCard.addSubview(detailLabel)
        contentCard.addSubview(phaseBadge)
        contentCard.addSubview(progressIndicator)
        contentCard.addSubview(progressValueLabel)
        contentCard.addSubview(footerLabel)
        contentCard.addSubview(openLogsButton)
        contentCard.addSubview(openDesktopButton)
        contentCard.addSubview(openBrowserButton)
        contentCard.addSubview(forceStopButton)

        NSLayoutConstraint.activate([
            contentCard.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 22),
            contentCard.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -22),
            contentCard.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 22),
            contentCard.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -22),

            eyebrowLabel.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 30),
            eyebrowLabel.topAnchor.constraint(equalTo: contentCard.topAnchor, constant: 24),

            iconPlate.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 30),
            iconPlate.topAnchor.constraint(equalTo: eyebrowLabel.bottomAnchor, constant: 18),
            iconPlate.widthAnchor.constraint(equalToConstant: 92),
            iconPlate.heightAnchor.constraint(equalToConstant: 92),

            iconView.leadingAnchor.constraint(equalTo: iconPlate.leadingAnchor, constant: 12),
            iconView.trailingAnchor.constraint(equalTo: iconPlate.trailingAnchor, constant: -12),
            iconView.topAnchor.constraint(equalTo: iconPlate.topAnchor, constant: 12),
            iconView.bottomAnchor.constraint(equalTo: iconPlate.bottomAnchor, constant: -12),

            titleLabel.leadingAnchor.constraint(equalTo: iconPlate.trailingAnchor, constant: 24),
            titleLabel.trailingAnchor.constraint(equalTo: contentCard.trailingAnchor, constant: -30),
            titleLabel.topAnchor.constraint(equalTo: iconPlate.topAnchor, constant: 2),

            detailLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            detailLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
            detailLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 8),

            phaseBadge.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            phaseBadge.topAnchor.constraint(equalTo: detailLabel.bottomAnchor, constant: 12),
            phaseBadge.heightAnchor.constraint(equalToConstant: 28),
            phaseBadge.widthAnchor.constraint(greaterThanOrEqualToConstant: 140),

            progressIndicator.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 30),
            progressIndicator.trailingAnchor.constraint(equalTo: progressValueLabel.leadingAnchor, constant: -16),
            progressBelowBadge,
            progressBelowIcon,
            progressIndicator.heightAnchor.constraint(equalToConstant: 20),

            progressValueLabel.trailingAnchor.constraint(equalTo: contentCard.trailingAnchor, constant: -30),
            progressValueLabel.centerYAnchor.constraint(equalTo: progressIndicator.centerYAnchor),
            progressValueLabel.widthAnchor.constraint(equalToConstant: 56),

            footerLabel.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 30),
            footerLabel.trailingAnchor.constraint(equalTo: contentCard.trailingAnchor, constant: -30),
            footerLabel.topAnchor.constraint(equalTo: progressIndicator.bottomAnchor, constant: 18),
            footerLabel.bottomAnchor.constraint(lessThanOrEqualTo: openLogsButton.topAnchor, constant: -16),

            openLogsButton.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 30),
            openLogsButton.bottomAnchor.constraint(equalTo: contentCard.bottomAnchor, constant: -24),

            openDesktopButton.leadingAnchor.constraint(equalTo: openLogsButton.trailingAnchor, constant: 10),
            openDesktopButton.bottomAnchor.constraint(equalTo: openLogsButton.bottomAnchor),

            openBrowserButton.leadingAnchor.constraint(equalTo: openDesktopButton.trailingAnchor, constant: 10),
            openBrowserButton.bottomAnchor.constraint(equalTo: openLogsButton.bottomAnchor),

            forceStopButton.trailingAnchor.constraint(equalTo: contentCard.trailingAnchor, constant: -30),
            forceStopButton.bottomAnchor.constraint(equalTo: openLogsButton.bottomAnchor),
        ])
    }

    private func resolvePythonExecutable() throws -> URL {
        let candidates = [
            auraRoot.appendingPathComponent(".venv/bin/python3"),
            URL(fileURLWithPath: "/opt/homebrew/bin/python3.12"),
            URL(fileURLWithPath: "/usr/local/bin/python3.12"),
            URL(fileURLWithPath: "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"),
        ]

        for candidate in candidates where fileManager.isExecutableFile(atPath: candidate.path) {
            return candidate
        }

        throw NSError(domain: "AuraLauncher", code: 5, userInfo: [
            NSLocalizedDescriptionKey: "Aura needs a Python 3.12 runtime, but the launcher could not find one.",
        ])
    }

    private func baseAuraEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let fallbackPath = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        if let currentPath = env["PATH"], !currentPath.isEmpty {
            env["PATH"] = currentPath + ":" + fallbackPath
        } else {
            env["PATH"] = fallbackPath
        }
        env["AURA_ATTACH_LAUNCHER"] = "0"
        env["AURA_LAUNCHED_FROM_APP"] = "1"
        env["AURA_SAFE_BOOT_DESKTOP"] = "1"
        env["AURA_EAGER_CORTEX_WARMUP"] = "0"
        env["AURA_DEFERRED_CORTEX_PREWARM"] = "0"
        env["AURA_SAFE_BOOT_METAL_CACHE_RATIO"] = "0.56"
        env["AURA_SAFE_BOOT_METAL_CACHE_CAP_GB"] = "36"
        env["PYTHONUNBUFFERED"] = "1"
        env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        env["OBJC_PRINT_LOAD_METHODS"] = "NO"
        return env
    }

    private func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private func requiresProtectedFolderFallback() -> Bool {
        let home = NSHomeDirectory()
        let protectedRoots = [
            "\(home)/Desktop",
            "\(home)/Documents",
            "\(home)/Downloads",
        ]
        return protectedRoots.contains { auraRoot.path.hasPrefix($0 + "/") || auraRoot.path == $0 }
    }

    private func terminalLaunchScriptURL() -> URL {
        lockDirectory.appendingPathComponent("desktop-terminal-launch.command")
    }

    private func spawnViaTerminal(arguments: [String]) throws {
        guard fileManager.fileExists(atPath: launchScript.path) else {
            throw NSError(domain: "AuraLauncher", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "launch_aura.sh is missing from the Aura repo.",
            ])
        }

        let pieces = [shellQuoted(launchScript.path)] + arguments.map(shellQuoted)
        let helperScript = """
        #!/bin/bash
        cd \(shellQuoted(auraRoot.path))
        export AURA_ATTACH_LAUNCHER=0
        export AURA_LAUNCHED_FROM_APP=1
        export AURA_SAFE_BOOT_DESKTOP=1
        export AURA_EAGER_CORTEX_WARMUP=0
        export AURA_DEFERRED_CORTEX_PREWARM=0
        export AURA_SAFE_BOOT_METAL_CACHE_RATIO=0.56
        export AURA_SAFE_BOOT_METAL_CACHE_CAP_GB=36
        \(pieces.joined(separator: " "))
        """
        let helperURL = terminalLaunchScriptURL()
        try helperScript.write(to: helperURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helperURL.path)
        writeTerminalHandoffMarker()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        proc.arguments = ["-a", "Terminal", helperURL.path]
        proc.currentDirectoryURL = auraRoot
        proc.environment = baseAuraEnvironment()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            throw NSError(
                domain: "AuraLauncher",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "Terminal handoff failed with status \(proc.terminationStatus)."]
            )
        }
    }


    private func spawnDetachedViaShell(arguments: [String]) throws {
        guard fileManager.fileExists(atPath: launchScript.path) else {
            throw NSError(domain: "AuraLauncher", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "launch_aura.sh is missing from the Aura repo.",
            ])
        }

        writeTerminalHandoffMarker()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = [launchScript.path] + arguments
        proc.currentDirectoryURL = auraRoot
        proc.environment = baseAuraEnvironment()
        let logHandle = try openLogHandle()
        proc.standardOutput = logHandle
        proc.standardError = logHandle
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            clearTerminalHandoffMarker()
            throw NSError(
                domain: "AuraLauncher",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "Launch helper failed with status \(proc.terminationStatus)."]
            )
        }
    }

    private func openLogHandle() throws -> FileHandle {
        let handle = try FileHandle(forWritingTo: logFile)
        try handle.seekToEnd()
        return handle
    }

    private func pollNow() {
        guard !isPolling else { return }
        isPolling = true

        let request = URLRequest(
            url: URL(string: "http://127.0.0.1:8000/api/health/boot")!,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: 1.0
        )

        session.dataTask(with: request) { [weak self] data, response, error in
            guard let self else { return }
            let snapshot = Self.parseSnapshot(data: data, response: response)
            DispatchQueue.main.async {
                self.isPolling = false
                self.handlePollResult(snapshot: snapshot, error: error)
            }
        }.resume()
    }

    private static func parseSnapshot(data: Data?, response: URLResponse?) -> BootSnapshot? {
        guard let http = response as? HTTPURLResponse else {
            return nil
        }

        var payload: [String: Any] = [:]
        if let data, !data.isEmpty,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            payload = json
        }

        return BootSnapshot(statusCode: http.statusCode, payload: payload)
    }

    private func handlePollResult(snapshot: BootSnapshot?, error: Error?) {
        if let snapshot {
            clearTerminalHandoffMarker()
            lastSnapshot = snapshot
            if let reason = snapshot.replacementReason(expectedSemver: bundledSemver),
               !launchInFlight,
               !forcedRelaunchAttempted {
                forcedRelaunchAttempted = true
                beginForcedRelaunch(reason: reason)
                return
            }
            renderSnapshot(snapshot)
            if snapshot.launcherReady {
                clearBootMarker()
                if autoOpenDesktopWindowIfNeeded() {
                    scheduleCloseIfNeeded()
                }
            }
            return
        }

        if bootMarkerIsStaleWithoutRuntime() {
            clearBootMarker()
        }
        if terminalHandoffIsStaleWithoutRuntime() {
            clearTerminalHandoffMarker()
        }

        if terminalHandoffIsFresh() {
            renderPendingLaunch(waitingOnExisting: true)
            footerLabel.stringValue = "Aura's launch helper has the handoff. Waiting for boot health from the live workspace."
            return
        }

        if bootMarkerIsFresh() {
            renderPendingLaunch(waitingOnExisting: true)
            return
        }

        if !launchInFlight {
            launchInFlight = true
            renderPendingLaunch(waitingOnExisting: false)
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                let result = self.launchAuraIfNeeded()
                DispatchQueue.main.async {
                    self.launchInFlight = false
                    switch result {
                    case .launched:
                        self.spawnedFreshRuntime = true
                        self.renderPendingLaunch(waitingOnExisting: true)
                    case .observingExistingBoot:
                        self.spawnedFreshRuntime = false
                        self.renderPendingLaunch(waitingOnExisting: true)
                    case .failed(let detail):
                        self.renderTitle("Aura hit a launch problem")
                        self.renderStatus(
                            detail: detail,
                            footer: "Open the logs for details, then try launching Aura again.",
                            progress: 0,
                            phase: "launch issue",
                            badgeStyle: .rose,
                        )
                    }
                }
            }
            return
        }

        renderPendingLaunch(waitingOnExisting: true)
        if let nsError = error as NSError?, nsError.code != NSURLErrorTimedOut {
            footerLabel.stringValue = "Aura hasn’t published boot health yet. Open the logs if this keeps happening."
        }
    }

    private func beginForcedRelaunch(reason: String) {
        launchInFlight = true
        renderTitle("Refreshing Aura’s live runtime")
        renderStatus(
            detail: "Replacing the older or unhealthy Aura runtime…",
            footer: reason,
            progress: 18.0,
            phase: "refreshing runtime",
            badgeStyle: .rose,
        )
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let result = self.launchAuraIfNeeded(forceRelaunch: true)
            DispatchQueue.main.async {
                self.launchInFlight = false
                switch result {
                case .launched:
                    self.spawnedFreshRuntime = true
                    self.renderPendingLaunch(waitingOnExisting: true)
                case .observingExistingBoot:
                    self.spawnedFreshRuntime = false
                    self.renderPendingLaunch(waitingOnExisting: true)
                case .failed(let detail):
                    self.renderTitle("Aura hit a launch problem")
                    self.renderStatus(
                        detail: detail,
                        footer: "Open the logs for details, then try launching Aura again.",
                        progress: 0,
                        phase: "launch issue",
                        badgeStyle: .rose,
                    )
                }
            }
        }
    }

    private func renderSnapshot(_ snapshot: BootSnapshot) {
        let normalized = snapshot.bootPhase.lowercased()
        let title: String
        let badgeStyle: BadgeStyle
        if normalized == "conversation_failed" {
            title = "Aura hit a local brain issue"
            badgeStyle = .rose
        } else if snapshot.launcherReady && !snapshot.conversationReady {
            title = "Aura is awake"
            badgeStyle = .emerald
        } else if snapshot.launcherReady {
            title = "Aura is ready"
            badgeStyle = .emerald
        } else if normalized == "conversation_warming" || normalized == "conversation_recovering" {
            title = "Aura is preparing the 32B lane"
            badgeStyle = .cyan
        } else {
            title = "Aura is booting"
            badgeStyle = .blue
        }

        let footer: String
        if normalized == "conversation_failed" {
            footer = "Aura’s core is online, but the local Cortex lane failed to start in this runtime. Open the logs for the exact backend error."
        } else if snapshot.launcherReady {
            footer = snapshot.conversationReady
                ? "Aura’s desktop window should appear momentarily."
                : "Aura’s core is online. The desktop window can open now while Cortex finishes recovering."
        } else if !snapshot.blockers.isEmpty {
            footer = "Boot phase: \(snapshot.phaseDisplay) • waiting on \(snapshot.blockers.joined(separator: ", "))"
        } else {
            footer = "Boot phase: \(snapshot.phaseDisplay)"
        }

        let progress = snapshot.launcherReady ? 100.0 : snapshot.progress

        renderTitle(title)
        renderStatus(
            detail: snapshot.statusMessage,
            footer: footer,
            progress: progress,
            phase: snapshot.phaseDisplay,
            badgeStyle: badgeStyle,
        )
    }

    private func renderPendingLaunch(waitingOnExisting: Bool) {
        let age = bootMarkerAge() ?? Date().timeIntervalSince(launchedAt)
        let progress = min(32.0, 10.0 + (age * 3.5))
        let terminalHandoffActive = terminalHandoffIsFresh()
        renderTitle("Aura is waking up")
        if waitingOnExisting {
            renderStatus(
                detail: terminalHandoffActive
                    ? "Launch handed off once. Waiting for Aura to publish boot health…"
                    : "Launch request sent. Waiting for Aura to publish boot health…",
                footer: terminalHandoffActive
                    ? "Aura’s launch helper is starting the live workspace. The launcher will not resend the request while that handoff is still fresh."
                    : "This window will update as soon as Aura’s kernel reports its boot phase.",
                progress: progress,
                phase: "waiting for health",
                badgeStyle: .cyan,
            )
        } else {
            renderStatus(
                detail: "Starting Aura’s desktop boot sequence…",
                footer: "The launcher will stay here until Aura is ready or shows a real boot phase.",
                progress: max(progress, 8.0),
                phase: "launch requested",
                badgeStyle: .violet,
            )
        }
    }

    private func renderTitle(_ text: String) {
        titleLabel.stringValue = text
    }

    private func renderStatus(
        detail: String,
        footer: String,
        progress: Double,
        phase: String,
        badgeStyle: BadgeStyle
    ) {
        detailLabel.stringValue = detail
        footerLabel.stringValue = footer
        progressIndicator.progress = max(0.0, min(100.0, progress))
        progressValueLabel.stringValue = "\(Int(progressIndicator.progress.rounded()))%"
        phaseBadge.stringValue = "  \(phase.uppercased())  "
        phaseBadge.textColor = badgeStyle.color
        phaseBadge.layer?.backgroundColor = badgeStyle.color.withAlphaComponent(0.16).cgColor
        phaseBadge.layer?.borderColor = badgeStyle.color.withAlphaComponent(0.34).cgColor
    }

    private func scheduleCloseIfNeeded() {
        guard !closeScheduled else { return }
        closeScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + readyCloseDelay) { [weak self] in
            NSApp.terminate(self)
        }
    }

    private func bootMarkerAge() -> TimeInterval? {
        guard let text = try? String(contentsOf: bootMarkerFile, encoding: .utf8) else {
            return nil
        }
        guard let epoch = Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return nil
        }
        return Date().timeIntervalSince1970 - epoch
    }

    private func bootMarkerIsFresh() -> Bool {
        guard let age = bootMarkerAge() else {
            return false
        }
        return age >= 0 && age < bootMarkerTTL
    }

    private func terminalHandoffAge() -> TimeInterval? {
        guard let text = try? String(contentsOf: terminalHandoffMarkerFile, encoding: .utf8) else {
            return nil
        }
        guard let epoch = Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return nil
        }
        return Date().timeIntervalSince1970 - epoch
    }

    private func terminalHandoffIsFresh() -> Bool {
        guard let age = terminalHandoffAge() else {
            return false
        }
        return age >= 0 && age < terminalHandoffWindow
    }

    private func runtimeLockFileURL() -> URL {
        lockDirectory.appendingPathComponent("orchestrator.lock")
    }

    private func runtimeLockIndicatesLiveProcess() -> Bool {
        let lockFile = runtimeLockFileURL()
        guard let text = try? String(contentsOf: lockFile, encoding: .utf8) else {
            return false
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let pid = Int32(trimmed), pid > 0 else {
            return false
        }
        return kill(pid, 0) == 0 || errno == EPERM
    }

    private func bootMarkerIsStaleWithoutRuntime() -> Bool {
        guard let age = bootMarkerAge(), age >= staleMarkerWithoutRuntimeWindow else {
            return false
        }
        return !runtimeLockIndicatesLiveProcess()
    }

    private func terminalHandoffIsStaleWithoutRuntime() -> Bool {
        guard let age = terminalHandoffAge(), age >= staleMarkerWithoutRuntimeWindow else {
            return false
        }
        return !runtimeLockIndicatesLiveProcess()
    }

    private func writeBootMarker() {
        let text = String(Date().timeIntervalSince1970)
        try? text.write(to: bootMarkerFile, atomically: true, encoding: .utf8)
    }

    private func clearBootMarker() {
        try? fileManager.removeItem(at: bootMarkerFile)
    }

    private func writeTerminalHandoffMarker() {
        let text = String(Date().timeIntervalSince1970)
        try? text.write(to: terminalHandoffMarkerFile, atomically: true, encoding: .utf8)
    }

    private func clearTerminalHandoffMarker() {
        try? fileManager.removeItem(at: terminalHandoffMarkerFile)
    }

    private func launchAuraIfNeeded(forceRelaunch: Bool = false) -> LaunchAttemptResult {
        if ProcessInfo.processInfo.environment["AURA_LAUNCHER_SKIP_SPAWN"] == "1" {
            return .observingExistingBoot
        }

        return withSpawnLock {
            if !forceRelaunch && self.bootMarkerIsStaleWithoutRuntime() {
                self.clearBootMarker()
            }
            if !forceRelaunch && self.terminalHandoffIsStaleWithoutRuntime() {
                self.clearTerminalHandoffMarker()
            }

            if !forceRelaunch && (self.bootMarkerIsFresh() || self.terminalHandoffIsFresh()) {
                return .observingExistingBoot
            }

            self.writeBootMarker()
            do {
                try self.spawnAuraProcess(forceRelaunch: forceRelaunch)
                return .launched
            } catch {
                self.clearBootMarker()
                self.clearTerminalHandoffMarker()
                return .failed(error.localizedDescription)
            }
        } ?? .observingExistingBoot
    }

    private func withSpawnLock(_ body: () -> LaunchAttemptResult) -> LaunchAttemptResult? {
        let fd = open(spawnLockFile.path, O_CREAT | O_RDWR, 0o644)
        guard fd != -1 else {
            return body()
        }
        defer { close(fd) }

        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            return nil
        }
        defer { flock(fd, LOCK_UN) }
        return body()
    }

    private func spawnAuraProcess(forceRelaunch: Bool = false) throws {
        guard fileManager.fileExists(atPath: auraMainScript.path) else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "aura_main.py is missing from the Aura repo.",
            ])
        }

        let directArguments = forceRelaunch
            ? ["-u", auraMainScript.path, "--desktop", "--reboot"]
            : ["-u", auraMainScript.path, "--desktop"]
        do {
            try spawnAuraSubprocess(arguments: directArguments)
        } catch {
            if requiresProtectedFolderFallback() {
                try spawnDetachedViaShell(arguments: forceRelaunch ? ["--reboot"] : [])
                return
            }
            throw error
        }
    }

    private func showFatalError(title: String, detail: String) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = title
        alert.informativeText = detail
        alert.addButton(withTitle: "Open Logs")
        alert.addButton(withTitle: "Close")

        if alert.runModal() == .alertFirstButtonReturn {
            openLogs()
        }
        NSApp.terminate(nil)
    }


    @objc private func forceStopAura() {
        renderTitle("Stopping Aura")
        renderStatus(
            detail: "Forcing Aura to stop and clearing launcher locks…",
            footer: "Use this if Aura gets stuck, keeps ports occupied, or won’t fully exit.",
            progress: 8.0,
            phase: "forcing stop",
            badgeStyle: .rose,
        )
        launchInFlight = true

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let detail = self.forceStopAuraProcess()
            DispatchQueue.main.async {
                self.launchInFlight = false
                self.clearBootMarker()
                self.clearTerminalHandoffMarker()
                self.lastSnapshot = nil
                self.forcedRelaunchAttempted = false
                self.autoDesktopOpenTriggered = false
                self.spawnedFreshRuntime = false
                self.renderTitle("Aura has been stopped")
                self.renderStatus(
                    detail: "Aura was force-stopped and the launcher reset its handoff state.",
                    footer: detail,
                    progress: 0.0,
                    phase: "stopped",
                    badgeStyle: .rose,
                )
            }
        }
    }

    private func forceStopAuraProcess() -> String {
        let cleanupScript = auraRoot.appendingPathComponent("aura_cleanup.py")
        let logHandle: FileHandle?
        do {
            logHandle = try openLogHandle()
        } catch {
            return "Aura stop failed before cleanup logging could start: \(error.localizedDescription)"
        }

        func runTool(arguments: [String]) -> Bool {
            let proc = Process()
            proc.executableURL = pythonExecutable
            proc.arguments = arguments
            proc.currentDirectoryURL = auraRoot
            proc.environment = baseAuraEnvironment()
            proc.standardOutput = logHandle
            proc.standardError = logHandle
            do {
                try proc.run()
                proc.waitUntilExit()
                return proc.terminationStatus == 0
            } catch {
                return false
            }
        }

        let stopOK = runTool(arguments: ["-u", auraMainScript.path, "--stop"])
        let cleanupOK = fileManager.fileExists(atPath: cleanupScript.path)
            ? runTool(arguments: [cleanupScript.path])
            : false

        if stopOK && cleanupOK {
            return "Aura’s runtime, workers, and stale locks were all cleared."
        }
        if cleanupOK {
            return "Aura needed the aggressive cleanup path, but the runtime and stale locks were cleared."
        }
        if stopOK {
            return "Aura’s main runtime stopped, but cleanup reported issues. Check the logs if ports still look busy."
        }
        return "The emergency stop path reported issues. Open the logs if Aura still appears to be running."
    }

    @objc private func openLogs() {
        NSWorkspace.shared.open(logFile)
    }

    @objc private func openDesktopWindow() {
        if let snapshot = lastSnapshot,
           snapshot.replacementReason(expectedSemver: bundledSemver) != nil,
           !launchInFlight {
            forcedRelaunchAttempted = false
            autoDesktopOpenTriggered = false
            beginForcedRelaunch(reason: "Refreshing Aura before opening the desktop window.")
            return
        }
        if terminalHandoffIsFresh() {
            footerLabel.stringValue = "Aura is already handling the desktop-window request. Give it a moment before trying again."
            return
        }
        do {
            try spawnAuxiliaryAura(arguments: ["--open-gui-window"])
            autoDesktopOpenTriggered = true
            NSApp.activate(ignoringOtherApps: true)
            NSRunningApplication.current.activate(options: [])
            NSApp.requestUserAttention(.informationalRequest)
        } catch {
            footerLabel.stringValue = "Aura’s desktop window could not be opened. Check the launcher logs."
        }
    }

    @objc private func openBrowser() {
        let build = bundledSemver.isEmpty ? "live" : bundledSemver
        let ts = Int(Date().timeIntervalSince1970)
        if let url = URL(string: "http://127.0.0.1:8000/?build=\(build)&ts=\(ts)") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func closeLauncher() {
        NSApp.terminate(nil)
    }

    private func spawnAuxiliaryAura(arguments: [String]) throws {
        let directArguments = ["-u", auraMainScript.path] + arguments
        do {
            try spawnAuraSubprocess(arguments: directArguments)
        } catch {
            if requiresProtectedFolderFallback() {
                if terminalHandoffIsFresh() {
                    return
                }
                try spawnDetachedViaShell(arguments: arguments)
                return
            }
            throw error
        }
    }

    private func spawnAuraSubprocess(arguments: [String]) throws {
        let proc = Process()
        proc.executableURL = pythonExecutable
        proc.arguments = arguments
        proc.currentDirectoryURL = auraRoot
        proc.environment = baseAuraEnvironment()
        let logHandle = try openLogHandle()
        proc.standardOutput = logHandle
        proc.standardError = logHandle
        try proc.run()
    }

    @discardableResult
    private func autoOpenDesktopWindowIfNeeded() -> Bool {
        if spawnedFreshRuntime {
            return true
        }
        if autoDesktopOpenTriggered {
            return true
        }
        if terminalHandoffIsFresh() {
            return true
        }
        do {
            try spawnAuxiliaryAura(arguments: ["--open-gui-window"])
            autoDesktopOpenTriggered = true
            NSApp.activate(ignoringOtherApps: true)
            NSRunningApplication.current.activate(options: [])
            NSApp.requestUserAttention(.informationalRequest)
            return true
        } catch {
            footerLabel.stringValue = "Aura is ready, but the desktop window didn’t open automatically. Use Open Desktop Window."
            return false
        }
    }
}

let app = NSApplication.shared
let delegate = AuraLauncherDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
