import AppKit
import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ScreenCaptureKit
import WebKit

private let nativeBridgeFlag = "--native-desktop-bridge"

private func bridgeJSON(_ payload: [String: Any], status: Int32 = 0) -> Never {
    let data = (try? JSONSerialization.data(withJSONObject: payload, options: [])) ?? Data("{\"ok\":false,\"error\":\"json_encoding_failed\"}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    Darwin.exit(status)
}

private func bridgeNumber(_ payload: [String: Any], _ key: String, default fallback: Double = 0) -> Double {
    if let value = payload[key] as? NSNumber { return value.doubleValue }
    if let value = payload[key] as? Double { return value }
    if let value = payload[key] as? Int { return Double(value) }
    return fallback
}

private func bridgeKeyCode(_ key: String) -> CGKeyCode? {
    let mapping: [String: CGKeyCode] = [
        "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
        "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
        "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22,
        "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29,
        "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "return": 36,
        "enter": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42,
        ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49,
        "`": 50, "backspace": 51, "delete": 51, "escape": 53, "esc": 53,
        "left": 123, "right": 124, "down": 125, "up": 126,
    ]
    return mapping[key.lowercased()]
}

private func bridgeModifierFlags(_ keys: [String]) -> CGEventFlags {
    var flags: CGEventFlags = []
    for key in keys.map({ $0.lowercased() }) {
        switch key {
        case "command", "cmd": flags.insert(.maskCommand)
        case "control", "ctrl": flags.insert(.maskControl)
        case "option", "alt": flags.insert(.maskAlternate)
        case "shift": flags.insert(.maskShift)
        default: break
        }
    }
    return flags
}

private func bridgePostKey(_ key: String, flags: CGEventFlags = []) -> Bool {
    guard let keyCode = bridgeKeyCode(key),
          let down = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: true),
          let up = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: false) else {
        return false
    }
    down.flags = flags
    up.flags = flags
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
    return true
}

private func bridgeAutomationProbe() -> [String: Any] {
    let source = """
    tell application "System Events"
        set frontName to name of first application process whose frontmost is true
        return frontName
    end tell
    """
    var errorInfo: NSDictionary?
    guard let script = NSAppleScript(source: source) else {
        return ["automation": false, "frontmost_app": "", "automation_error": "script_compile_failed"]
    }
    let descriptor = script.executeAndReturnError(&errorInfo)
    if let errorInfo {
        let message = String(describing: errorInfo)
        return [
            "automation": false,
            "frontmost_app": "",
            "automation_error": message.prefix(240),
        ]
    }
    return [
        "automation": true,
        "frontmost_app": descriptor.stringValue ?? "",
    ]
}

private let bridgeScreenCapturePolicySchema = "aura.security.screen_capture_privacy_policy.v1"

private struct BridgeScreenCapturePrivacyPolicyFile: Decodable {
    let schema: String
    let privateWindowMarkers: [String]
    let privateApps: [String]
    let privateBrowsingApps: [String]

    enum CodingKeys: String, CodingKey {
        case schema
        case privateWindowMarkers = "private_window_markers"
        case privateApps = "private_apps"
        case privateBrowsingApps = "private_browsing_apps"
    }
}

private struct BridgeScreenCapturePrivacyPolicy {
    let privateWindowMarkers: [String]
    let privateApps: Set<String>
    let privateBrowsingApps: Set<String>
}

private struct BridgeScreenCaptureAdmission {
    let allowed: Bool
    let reason: String
    let contextKnown: Bool

    var receipt: [String: Any] {
        [
            "schema": "aura.security.screen_capture_admission.v1",
            "allowed": allowed,
            "reason": reason,
            "context_known": contextKnown,
            "authority": "resident_bridge",
        ]
    }
}

private struct BridgeForegroundWindowContext {
    let appName: String
    let processIdentifier: pid_t
    let title: String
    let windowID: CGWindowID
    let bounds: CGRect
    let windows: [[String: Any]]
}

private let bridgeFrameSequenceLock = NSLock()
private var bridgeFrameSequence: UInt64 = 0

private func nextBridgeFrameSequence() -> UInt64 {
    bridgeFrameSequenceLock.lock()
    defer { bridgeFrameSequenceLock.unlock() }
    bridgeFrameSequence &+= 1
    return bridgeFrameSequence
}

private func bridgeScreenSessionLocked() -> Bool {
    guard let session = CGSessionCopyCurrentDictionary() as? [String: Any] else {
        // A missing session dictionary cannot authorize pixels from whatever
        // may be behind loginwindow or a fast-user-switch boundary.
        return true
    }
    if let locked = session["CGSSessionScreenIsLocked"] as? NSNumber,
       locked.boolValue {
        return true
    }

    // macOS omits CGSSessionScreenIsLocked while an ordinary session is
    // unlocked. Positive console ownership and completed-login evidence are
    // therefore the authority to proceed; either missing signal still fails
    // closed. Treating the absent lock key itself as locked made the resident
    // privacy bridge permanently return foreground_unknown on every healthy
    // desktop session.
    guard let onConsole = session["kCGSSessionOnConsoleKey"] as? NSNumber,
          onConsole.boolValue,
          let loginDone = session["kCGSessionLoginDoneKey"] as? NSNumber,
          loginDone.boolValue else {
        return true
    }
    return false
}

private func bridgeForegroundWindowContext() -> BridgeForegroundWindowContext? {
    guard !bridgeScreenSessionLocked() else {
        return nil
    }
    guard let application = NSWorkspace.shared.frontmostApplication else {
        return nil
    }
    let appName = (application.localizedName ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !appName.isEmpty else {
        return nil
    }

    let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
    let windows = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        as? [[String: Any]] ?? []
    func visibleWindow(
        ownedBy requestedPID: pid_t? = nil
    ) -> (appName: String, processIdentifier: pid_t, title: String, windowID: CGWindowID, bounds: CGRect)? {
        for window in windows {
        let ownerPID = (window[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value ?? 0
        let layer = (window[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 0
            guard layer == 0, ownerPID > 0 else { continue }
            if let requestedPID, ownerPID != requestedPID { continue }
            if requestedPID == nil {
                guard let ownerApplication = NSRunningApplication(
                    processIdentifier: ownerPID
                ), ownerApplication.activationPolicy == .regular else {
                    continue
                }
            }
            let windowID = (window[kCGWindowNumber as String] as? NSNumber)?.uint32Value ?? 0
            guard windowID != 0,
                  let rawBounds = window[kCGWindowBounds as String] as? [String: Any],
                  let bounds = CGRect(
                      dictionaryRepresentation: rawBounds as CFDictionary
                  ), bounds.width > 1, bounds.height > 1 else {
                continue
            }
            let title = (window[kCGWindowName as String] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let ownerName = (window[kCGWindowOwnerName as String] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedName = NSRunningApplication(processIdentifier: ownerPID)?
                .localizedName?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? ownerName
            guard !resolvedName.isEmpty else { continue }
            return (resolvedName, ownerPID, title, windowID, bounds)
        }
        return nil
    }

    // NSWorkspace can transiently report a menu-only or system process as
    // frontmost. In that case the front-to-back CoreGraphics order supplies
    // the actual topmost regular application window.
    guard let selected = visibleWindow(ownedBy: application.processIdentifier)
        ?? visibleWindow() else {
        return nil
    }
    return BridgeForegroundWindowContext(
        appName: selected.appName.isEmpty ? appName : selected.appName,
        processIdentifier: selected.processIdentifier,
        title: selected.title,
        windowID: selected.windowID,
        bounds: selected.bounds,
        windows: windows
    )
}

private let bridgeScreenCapturePrivacyPolicy: BridgeScreenCapturePrivacyPolicy? = {
    guard let resource = Bundle.main.url(
        forResource: "screen_capture_privacy_policy",
        withExtension: "json"
    ), let data = try? Data(contentsOf: resource),
       let decoded = try? JSONDecoder().decode(
           BridgeScreenCapturePrivacyPolicyFile.self,
           from: data
       ), decoded.schema == bridgeScreenCapturePolicySchema else {
        return nil
    }

    func normalized(_ values: [String]) -> [String]? {
        let result = values.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        guard !result.isEmpty,
              !result.contains(where: { $0.isEmpty }),
              Set(result).count == result.count else {
            return nil
        }
        return result
    }

    guard let markers = normalized(decoded.privateWindowMarkers),
          let privateApps = normalized(decoded.privateApps),
          let privateBrowsingApps = normalized(decoded.privateBrowsingApps) else {
        return nil
    }
    return BridgeScreenCapturePrivacyPolicy(
        privateWindowMarkers: markers,
        privateApps: Set(privateApps),
        privateBrowsingApps: Set(privateBrowsingApps)
    )
}()

private func bridgeScreenCaptureAdmission(
    context suppliedContext: BridgeForegroundWindowContext? = nil
) -> BridgeScreenCaptureAdmission {
    guard let policy = bridgeScreenCapturePrivacyPolicy else {
        return BridgeScreenCaptureAdmission(
            allowed: false,
            reason: "policy_unavailable",
            contextKnown: false
        )
    }
    guard let context = suppliedContext ?? bridgeForegroundWindowContext() else {
        return BridgeScreenCaptureAdmission(
            allowed: false,
            reason: "foreground_unknown",
            contextKnown: false
        )
    }
    let loweredApp = context.appName.lowercased()
    for window in context.windows {
        let layer = (window[kCGWindowLayer as String] as? NSNumber)?.intValue ?? 0
        guard layer == 0 else { continue }
        let ownerPID = (window[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value ?? 0
        let owner = (window[kCGWindowOwnerName as String] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let windowTitle = (window[kCGWindowName as String] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let loweredOwner = owner.lowercased()
        let combined = "\(owner) \(windowTitle)".lowercased()
        if policy.privateApps.contains(loweredOwner)
            || policy.privateWindowMarkers.contains(where: combined.contains) {
            return BridgeScreenCaptureAdmission(
                allowed: false,
                reason: ownerPID == context.processIdentifier
                    ? "private_foreground"
                    : "private_visible",
                contextKnown: true
            )
        }
        if policy.privateBrowsingApps.contains(loweredOwner) && windowTitle.isEmpty {
            return BridgeScreenCaptureAdmission(
                allowed: false,
                reason: "browser_title_unknown",
                contextKnown: false
            )
        }
    }
    if policy.privateBrowsingApps.contains(loweredApp) && context.title.isEmpty {
        return BridgeScreenCaptureAdmission(
            allowed: false,
            reason: "browser_title_unknown",
            contextKnown: false
        )
    }
    return BridgeScreenCaptureAdmission(
        allowed: true,
        reason: "none",
        contextKnown: true
    )
}

private func bridgeScreenCaptureRefusal(
    _ admission: BridgeScreenCaptureAdmission
) -> ([String: Any], Int32) {
    return ([
        "ok": false,
        "error": "screen_capture_refused",
        "capture_admission": admission.receipt,
    ], 2)
}

private func bridgeForegroundContextMatches(
    _ before: BridgeForegroundWindowContext,
    _ after: BridgeForegroundWindowContext
) -> Bool {
    before.processIdentifier == after.processIdentifier
        && before.windowID == after.windowID
        && before.title == after.title
        && before.bounds.equalTo(after.bounds)
}

private final class BridgeCapturedImageBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.aura.desktop.capture-result")
    private var image: CGImage?

    func store(_ value: CGImage) {
        queue.sync { image = value }
    }

    func load() -> CGImage? {
        queue.sync { image }
    }
}

private func bridgeCaptureWindowImage(windowID: CGWindowID) -> CGImage? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    let semaphore = DispatchSemaphore(value: 0)
    let result = BridgeCapturedImageBox()
    Task {
        defer { semaphore.signal() }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                true,
                onScreenWindowsOnly: true
            )
            guard let window = content.windows.first(where: {
                $0.windowID == windowID
            }) else {
                return
            }
            let configuration = SCStreamConfiguration()
            configuration.width = max(1, Int(window.frame.width))
            configuration.height = max(1, Int(window.frame.height))
            configuration.showsCursor = false
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: configuration
            )
            result.store(image)
        } catch {
            return
        }
    }
    guard semaphore.wait(timeout: .now() + 8.0) == .success else {
        return nil
    }
    return result.load()
}

private func bridgeObserveForegroundFrame() -> ([String: Any], Int32) {
    guard let before = bridgeForegroundWindowContext() else {
        return bridgeScreenCaptureRefusal(bridgeScreenCaptureAdmission())
    }
    let admission = bridgeScreenCaptureAdmission(context: before)
    guard admission.allowed else {
        return bridgeScreenCaptureRefusal(admission)
    }
    guard let image = bridgeCaptureWindowImage(windowID: before.windowID) else {
        return (["ok": false, "error": "screen_capture_unavailable"], 2)
    }
    let representation = NSBitmapImageRep(cgImage: image)
    guard let png = representation.representation(using: .png, properties: [:]) else {
        return (["ok": false, "error": "screen_capture_encoding_failed"], 2)
    }

    guard let after = bridgeForegroundWindowContext() else {
        return (["ok": false, "error": "foreground_changed"], 2)
    }
    let finalAdmission = bridgeScreenCaptureAdmission(context: after)
    guard finalAdmission.allowed else {
        return bridgeScreenCaptureRefusal(finalAdmission)
    }
    guard bridgeForegroundContextMatches(before, after) else {
        return (["ok": false, "error": "foreground_changed"], 2)
    }

    let digest = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
    let contextRevision = [
        String(before.processIdentifier),
        String(before.windowID),
        String(Int(before.bounds.origin.x)),
        String(Int(before.bounds.origin.y)),
        String(Int(before.bounds.width)),
        String(Int(before.bounds.height)),
        before.title,
    ].joined(separator: ":")
    return ([
        "ok": true,
        "schema": "aura.perception.foreground_frame.v1",
        "sequence": nextBridgeFrameSequence(),
        "captured_monotonic_ns": DispatchTime.now().uptimeNanoseconds,
        "context_revision": contextRevision,
        "app": before.appName,
        "title": before.title,
        "window_id": before.windowID,
        "bounds": [
            "x": before.bounds.origin.x,
            "y": before.bounds.origin.y,
            "width": before.bounds.width,
            "height": before.bounds.height,
        ],
        "width": image.width,
        "height": image.height,
        "byte_length": png.count,
        "frame_sha256": digest,
        "frame_base64": png.base64EncodedString(),
        "capture_admission": finalAdmission.receipt,
    ], 0)
}

private func bridgeActivateForPermissionPrompt() {
    NSRunningApplication.current.activate(options: [.activateAllWindows])
}

private func bridgeTypeText(_ text: String, interval: TimeInterval) -> Bool {
    for scalar in text.unicodeScalars {
        var unit = UniChar(scalar.value)
        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true),
              let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false) else {
            return false
        }
        down.keyboardSetUnicodeString(stringLength: 1, unicodeString: &unit)
        up.keyboardSetUnicodeString(stringLength: 1, unicodeString: &unit)
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        if interval > 0 { Thread.sleep(forTimeInterval: interval) }
    }
    return true
}

private func bridgeMouseButton(_ name: String) -> (CGMouseButton, CGEventType, CGEventType) {
    switch name.lowercased() {
    case "right": return (.right, .rightMouseDown, .rightMouseUp)
    case "middle": return (.center, .otherMouseDown, .otherMouseUp)
    default: return (.left, .leftMouseDown, .leftMouseUp)
    }
}

private func nativeDesktopBridgeResult(payload: [String: Any]) -> ([String: Any], Int32) {
    let command = String(describing: payload["command"] ?? "probe").lowercased()
    let displayID = CGMainDisplayID()
    let width = Int(CGDisplayPixelsWide(displayID))
    let height = Int(CGDisplayPixelsHigh(displayID))

    switch command {
    case "probe":
        var response: [String: Any] = [
            "ok": true,
            "screen_recording": CGPreflightScreenCaptureAccess(),
            "accessibility": AXIsProcessTrusted(),
            "bundle_identifier": Bundle.main.bundleIdentifier ?? "",
            "width": width,
            "height": height,
        ]
        for (key, value) in bridgeAutomationProbe() {
            response[key] = value
        }
        return (response, 0)
    case "request_screen":
        bridgeActivateForPermissionPrompt()
        let granted = CGRequestScreenCaptureAccess()
        return ([
            "ok": granted,
            "screen_recording": granted,
            "bundle_identifier": Bundle.main.bundleIdentifier ?? "",
            "width": width,
            "height": height,
        ], granted ? 0 : 2)
    case "request_accessibility":
        bridgeActivateForPermissionPrompt()
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        let granted = AXIsProcessTrustedWithOptions(options)
        return ([
            "ok": granted,
            "accessibility": granted,
            "bundle_identifier": Bundle.main.bundleIdentifier ?? "",
            "width": width,
            "height": height,
        ], granted ? 0 : 2)
    case "size":
        return (["ok": true, "width": width, "height": height], 0)
    case "position":
        let location = CGEvent(source: nil)?.location ?? .zero
        return (["ok": true, "x": location.x, "y": location.y], 0)
    case "foreground_capture_admission":
        let admission = bridgeScreenCaptureAdmission()
        return ([
            "ok": true,
            "capture_admission": admission.receipt,
        ], 0)
    case "frontmost_window_context":
        guard let context = bridgeForegroundWindowContext() else {
            return bridgeScreenCaptureRefusal(bridgeScreenCaptureAdmission())
        }
        let admission = bridgeScreenCaptureAdmission(context: context)
        guard admission.allowed else {
            return bridgeScreenCaptureRefusal(admission)
        }
        return ([
            "ok": true,
            "app": context.appName,
            "title": context.title,
            "window_id": context.windowID,
            "capture_admission": admission.receipt,
        ], 0)
    case "observe_foreground_frame":
        return bridgeObserveForegroundFrame()
    case "screenshot":
        let admission = bridgeScreenCaptureAdmission()
        guard admission.allowed else {
            return bridgeScreenCaptureRefusal(admission)
        }
        guard let path = payload["path"] as? String, !path.isEmpty else {
            return (["ok": false, "error": "screen_capture_unavailable"], 2)
        }
        let target = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let capture = Process()
        capture.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        capture.arguments = ["-x", "-t", "png", path]
        capture.standardOutput = FileHandle.nullDevice
        capture.standardError = FileHandle.nullDevice
        let finalAdmission = bridgeScreenCaptureAdmission()
        guard finalAdmission.allowed else {
            return bridgeScreenCaptureRefusal(finalAdmission)
        }
        do {
            try capture.run()
            capture.waitUntilExit()
        } catch {
            return (["ok": false, "error": "screen_capture_launch_failed"], 2)
        }
        guard capture.terminationStatus == 0, FileManager.default.fileExists(atPath: path) else {
            return (["ok": false, "error": "screen_capture_write_failed"], 2)
        }
        return (["ok": true, "path": path, "width": width, "height": height], 0)
    case "move":
        let point = CGPoint(x: bridgeNumber(payload, "x"), y: bridgeNumber(payload, "y"))
        guard let event = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left) else {
            return (["ok": false, "error": "mouse_event_unavailable"], 2)
        }
        event.post(tap: .cghidEventTap)
        return (["ok": true, "x": point.x, "y": point.y], 0)
    case "click":
        let current = CGEvent(source: nil)?.location ?? .zero
        let point = CGPoint(
            x: bridgeNumber(payload, "x", default: current.x),
            y: bridgeNumber(payload, "y", default: current.y)
        )
        let buttonName = String(describing: payload["button"] ?? "left")
        let (button, downType, upType) = bridgeMouseButton(buttonName)
        let clicks = max(1, Int(bridgeNumber(payload, "clicks", default: 1)))
        let interval = max(0, bridgeNumber(payload, "interval"))
        for clickIndex in 1...clicks {
            guard let down = CGEvent(mouseEventSource: nil, mouseType: downType, mouseCursorPosition: point, mouseButton: button),
                  let up = CGEvent(mouseEventSource: nil, mouseType: upType, mouseCursorPosition: point, mouseButton: button) else {
                return (["ok": false, "error": "mouse_event_unavailable"], 2)
            }
            down.setIntegerValueField(.mouseEventClickState, value: Int64(clickIndex))
            up.setIntegerValueField(.mouseEventClickState, value: Int64(clickIndex))
            down.post(tap: .cghidEventTap)
            up.post(tap: .cghidEventTap)
            if interval > 0 { Thread.sleep(forTimeInterval: interval) }
        }
        return (["ok": true, "x": point.x, "y": point.y, "clicks": clicks], 0)
    case "write":
        let text = String(describing: payload["text"] ?? "")
        let ok = bridgeTypeText(text, interval: max(0, bridgeNumber(payload, "interval")))
        return (["ok": ok, "characters": text.count], ok ? 0 : 2)
    case "press":
        let key = String(describing: payload["key"] ?? "")
        let presses = max(1, Int(bridgeNumber(payload, "presses", default: 1)))
        let interval = max(0, bridgeNumber(payload, "interval"))
        for _ in 0..<presses {
            guard bridgePostKey(key) else {
                return (["ok": false, "error": "unsupported_key", "key": key], 2)
            }
            if interval > 0 { Thread.sleep(forTimeInterval: interval) }
        }
        return (["ok": true, "key": key, "presses": presses], 0)
    case "hotkey":
        let keys = (payload["keys"] as? [String]) ?? []
        guard let finalKey = keys.last else {
            return (["ok": false, "error": "hotkey_requires_key"], 2)
        }
        let ok = bridgePostKey(finalKey, flags: bridgeModifierFlags(Array(keys.dropLast())))
        return (["ok": ok, "keys": keys], ok ? 0 : 2)
    case "scroll":
        let amount = Int32(bridgeNumber(payload, "amount"))
        guard let event = CGEvent(
            scrollWheelEvent2Source: nil,
            units: .line,
            wheelCount: 1,
            wheel1: amount,
            wheel2: 0,
            wheel3: 0
        ) else {
            return (["ok": false, "error": "scroll_event_unavailable"], 2)
        }
        event.post(tap: .cghidEventTap)
        return (["ok": true, "amount": amount], 0)
    default:
        return (["ok": false, "error": "unsupported_command", "command": command], 2)
    }
}

private func nativeDesktopBridgeCommandRequiresMainThread(payload: [String: Any]) -> Bool {
    let command = String(describing: payload["command"] ?? "probe").lowercased()
    return command == "request_screen" || command == "request_accessibility"
}

private func evaluateNativeDesktopBridge(payload: [String: Any]) -> ([String: Any], Int32) {
    if Thread.isMainThread || !nativeDesktopBridgeCommandRequiresMainThread(payload: payload) {
        return nativeDesktopBridgeResult(payload: payload)
    }
    var bridgedResult: ([String: Any], Int32) = (["ok": false, "error": "main_thread_bridge_uninitialized"], 2)
    DispatchQueue.main.sync {
        bridgedResult = nativeDesktopBridgeResult(payload: payload)
    }
    return bridgedResult
}

private func runNativeDesktopBridge(payload: [String: Any]) -> Never {
    let (result, status) = evaluateNativeDesktopBridge(payload: payload)
    bridgeJSON(result, status: status)
}

private let pollInterval: TimeInterval = 0.8
private let bootMarkerTTL: TimeInterval = 180.0
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
        return (checks["runtime_integrity"] as? Bool) ?? true
    }

    var checks: [String: Any] {
        (payload["checks"] as? [String: Any]) ?? [:]
    }

    var runtimeLoopRunning: Bool {
        (checks["running"] as? Bool) ?? true
    }

    var runtimeContractHealthy: Bool {
        (checks["runtime_contract_healthy"] as? Bool) ?? true
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

    var conversationOperational: Bool {
        if conversationReady {
            return true
        }
        if let lane = payload["conversation_lane"] as? [String: Any] {
            let state = String(describing: lane["state"] ?? "").lowercased()
            if [
                "ready",
                "serving",
                "working",
                "generating",
                "busy",
                "foreground_generation",
                "handshaking",
            ].contains(state) {
                return true
            }
            if (lane["warmup_in_flight"] as? Bool) == true {
                return true
            }
        }
        let normalized = bootPhase.lowercased()
        return [
            "conversation_operational",
            "conversation_working",
            "kernel_ready",
            "proxy_ready",
        ].contains(normalized)
    }

    var runtimeHasUserVisibleHandoff: Bool {
        launcherReady || systemReady || conversationOperational
    }

    var blockers: [String] {
        (payload["blockers"] as? [String]) ?? []
    }

    var phaseDisplay: String {
        bootPhase.replacingOccurrences(of: "_", with: " ")
    }

    var hasDeadMindTickBlocker: Bool {
        blockers.contains { blocker in
            blocker == "important:mind_tick" || blocker == "contract/important:mind_tick"
        }
    }

    var hasEventLoopMonitorBlocker: Bool {
        blockers.contains { blocker in
            blocker == "important:event_loop_monitor" || blocker == "contract/important:event_loop_monitor"
        }
    }

    var staleRuntimeFailureReason: String? {
        let normalized = bootPhase.lowercased()
        if runtimeAge >= 45.0 && !runtimeLoopRunning {
            return "Existing runtime lock points at a process whose boot loop is no longer running."
        }
        // A dead mind tick only warrants replacement when the runtime is NOT
        // actually serving the user (a true zombie: foreground looks open but
        // cognition is gone). If the conversation lane / system is live
        // (runtimeHasUserVisibleHandoff), a transient background-loop stall — e.g.
        // the periodic dream-consolidation freeze, or a memory-backpressured tick —
        // must surface in telemetry, NOT SIGTERM a runtime the user can still talk
        // to and trigger a respawn during the slow 32B warm. Kernel death itself is
        // caught by the REQUIRED kernel probe. (2026-07: this line replaced healthy
        // runtimes on a transient false-death → serial respawn.)
        if runtimeAge >= 90.0 && hasDeadMindTickBlocker && !runtimeHasUserVisibleHandoff {
            return "Existing runtime has a dead mind tick and no live conversation lane; replacing the zombie session."
        }

        // After Aura is openable or the conversation lane is doing real work,
        // the launcher is an observer. Core-loop degradation must surface in
        // health/neural telemetry, not SIGTERM the user's active session. The
        // explicit Force Stop button remains the operator-owned escape hatch.
        // A dead mind tick is the exception: it means the foreground can look
        // open while the canonical cognition loop is already gone.
        if runtimeHasUserVisibleHandoff {
            return nil
        }
        if runtimeAge >= 90.0 && (hasDeadMindTickBlocker || hasEventLoopMonitorBlocker) {
            return "Existing runtime is explicitly blocked by a dead core loop."
        }
        if runtimeAge >= 90.0 && normalized == "kernel_warming" && !runtimeContractHealthy {
            return "Existing runtime stayed in kernel warming with an unhealthy runtime contract."
        }
        return nil
    }

    func replacementReason(expectedSemver: String) -> String? {
        let trimmedExpected = expectedSemver.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedExpected = trimmedExpected.split(separator: "-", maxSplits: 1).first.map(String.init) ?? trimmedExpected
        let normalizedServed = semver.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "-", maxSplits: 1).first.map(String.init) ?? semver
        if !normalizedExpected.isEmpty, !normalizedServed.isEmpty, normalizedServed != normalizedExpected {
            return "Existing runtime is serving build \(semver), but launcher expects \(trimmedExpected)."
        }

        if let staleReason = staleRuntimeFailureReason {
            return staleReason
        }

        // Once the kernel has completed handoff, the launcher is an observer.
        // A foreground conversation can transiently return 5xx while Cortex
        // resets or a quality gate retries; converting that into --reboot
        // destroys the user's session and the model's warm state. Runtime
        // recovery owns post-handoff failures. Forced replacement remains
        // available through the explicit UI action.
        if runtimeHasUserVisibleHandoff {
            return nil
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
    static let auraCanvasTop = NSColor(calibratedRed: 0.08, green: 0.09, blue: 0.15, alpha: 1.0)
    static let auraCanvasBottom = NSColor(calibratedRed: 0.03, green: 0.04, blue: 0.08, alpha: 1.0)
    static let auraPanel = NSColor(calibratedRed: 0.10, green: 0.11, blue: 0.18, alpha: 0.90)
    static let auraPanelBorder = NSColor(calibratedRed: 0.40, green: 0.32, blue: 0.80, alpha: 0.34)
    static let auraTrack = NSColor(calibratedRed: 0.16, green: 0.18, blue: 0.25, alpha: 1.0)
    static let auraCyan = NSColor(calibratedRed: 0.18, green: 0.86, blue: 1.0, alpha: 1.0)
    static let auraBlue = NSColor(calibratedRed: 0.18, green: 0.49, blue: 1.0, alpha: 1.0)
    static let auraViolet = NSColor(calibratedRed: 0.61, green: 0.36, blue: 1.0, alpha: 1.0)
    static let auraAmber = NSColor(calibratedRed: 1.0, green: 0.73, blue: 0.30, alpha: 1.0)
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

        let middleColor = NSColor(calibratedRed: 0.05, green: 0.05, blue: 0.12, alpha: 1.0)
        gradientLayer.colors = [NSColor.auraCanvasTop.cgColor, middleColor.cgColor, NSColor.auraCanvasBottom.cgColor]
        gradientLayer.locations = [0.0, 0.5, 1.0]
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
        trackLayer.cornerRadius = 4
        layer?.addSublayer(trackLayer)

        fillGlowLayer.backgroundColor = NSColor.auraBlue.withAlphaComponent(0.55).cgColor
        fillGlowLayer.cornerRadius = 4
        fillGlowLayer.shadowColor = NSColor.auraViolet.cgColor
        fillGlowLayer.shadowOpacity = 0.9
        fillGlowLayer.shadowRadius = 8
        fillGlowLayer.shadowOffset = .zero
        layer?.addSublayer(fillGlowLayer)

        fillGradientLayer.colors = [
            NSColor.auraCyan.cgColor,
            NSColor.auraBlue.cgColor,
            NSColor.auraViolet.cgColor,
        ]
        fillGradientLayer.startPoint = CGPoint(x: 0.0, y: 0.5)
        fillGradientLayer.endPoint = CGPoint(x: 1.0, y: 0.5)
        fillGradientLayer.cornerRadius = 4
        layer?.addSublayer(fillGradientLayer)

        let glowPulse = CABasicAnimation(keyPath: "shadowOpacity")
        glowPulse.fromValue = 0.55
        glowPulse.toValue = 0.95
        glowPulse.duration = 1.8
        glowPulse.autoreverses = true
        glowPulse.repeatCount = .infinity
        glowPulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        fillGlowLayer.add(glowPulse, forKey: "glowPulse")
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        let rect = bounds
        trackLayer.frame = rect

        let clamped = max(0.0, min(100.0, progress))
        let fillWidth = max(8.0, rect.width * CGFloat(clamped / 100.0))
        let fillRect = CGRect(x: rect.minX, y: rect.minY, width: min(fillWidth, rect.width), height: rect.height)

        fillGlowLayer.isHidden = clamped <= 0.0
        fillGradientLayer.isHidden = clamped <= 0.0
        fillGlowLayer.frame = fillRect
        fillGradientLayer.frame = fillRect
    }
}

private final class LauncherChipLabel: NSView {
    enum Tone {
        case cyan
        case violet
        case neutral

        var foreground: NSColor {
            switch self {
            case .cyan:
                return NSColor.auraCyan
            case .violet:
                return NSColor.auraViolet
            case .neutral:
                return NSColor(calibratedWhite: 0.92, alpha: 1.0)
            }
        }

        var background: NSColor {
            switch self {
            case .cyan:
                return NSColor.auraCyan.withAlphaComponent(0.10)
            case .violet:
                return NSColor.auraViolet.withAlphaComponent(0.12)
            case .neutral:
                return NSColor.white.withAlphaComponent(0.06)
            }
        }

        var border: NSColor {
            switch self {
            case .cyan:
                return NSColor.auraCyan.withAlphaComponent(0.26)
            case .violet:
                return NSColor.auraViolet.withAlphaComponent(0.26)
            case .neutral:
                return NSColor.white.withAlphaComponent(0.10)
            }
        }
    }

    private let textField = NSTextField(labelWithString: "")

    init(_ text: String, tone: Tone) {
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false
        wantsLayer = true
        layer?.backgroundColor = tone.background.cgColor
        layer?.borderColor = tone.border.cgColor
        layer?.borderWidth = 1.0
        layer?.cornerRadius = 8

        textField.stringValue = text.uppercased()
        textField.translatesAutoresizingMaskIntoConstraints = false
        textField.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .bold)
        textField.textColor = tone.foreground
        textField.alignment = .center
        textField.drawsBackground = false
        textField.isBordered = false
        textField.isSelectable = false
        addSubview(textField)

        NSLayoutConstraint.activate([
            textField.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            textField.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            textField.topAnchor.constraint(equalTo: topAnchor, constant: 4),
            textField.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -4)
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

/// Scanline-cut wordmark, matching the web splash's treatment: the glyphs are
/// filled with horizontal stripes and bloomed with a layered neon glow.
///
/// The striped glyphs are composited offscreen first, then drawn into the view
/// through `setShadow`. Drawing stripes directly under a text clip would confine
/// the shadow to the clip region and kill the bloom — the glow has to come from
/// an already-striped image.
private final class ScanlineWordmarkView: NSView {
    var text: String { didSet { invalidateCache() } }
    var tint: NSColor { didSet { invalidateCache() } }
    var glow: NSColor { didSet { invalidateCache() } }
    var pointSize: CGFloat { didSet { invalidateCache() } }

    /// Stripe pitch as a fraction of point size, so it tracks the type like the
    /// web build's `em`-based pitch.
    private let stripeRatio: CGFloat = 0.055
    private var cachedImage: NSImage?
    private var cachedScale: CGFloat = 0

    init(text: String, tint: NSColor, glow: NSColor, pointSize: CGFloat) {
        self.text = text
        self.tint = tint
        self.glow = glow
        self.pointSize = pointSize
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false
        wantsLayer = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func invalidateCache() {
        cachedImage = nil
        invalidateIntrinsicContentSize()
        needsDisplay = true
    }

    private var font: NSFont {
        // Avenir Next Heavy reads closest to the web build's Syne 900; the
        // system black weight is a dependable fallback on any macOS.
        NSFont(name: "AvenirNextCondensed-Heavy", size: pointSize)
            ?? NSFont.systemFont(ofSize: pointSize, weight: .black)
    }

    private var attributes: [NSAttributedString.Key: Any] {
        [
            .font: font,
            .kern: pointSize * 0.02,
            .foregroundColor: NSColor.white,
        ]
    }

    override var intrinsicContentSize: NSSize {
        let size = (text as NSString).size(withAttributes: attributes)
        // Room for the bloom so it is never clipped by the view bounds.
        return NSSize(width: ceil(size.width) + pointSize * 0.5,
                      height: ceil(size.height) + pointSize * 0.35)
    }

    /// White glyphs -> stripe clip -> tinted stripes, as a standalone image.
    private func stripedImage(scale: CGFloat) -> NSImage? {
        let size = intrinsicContentSize
        guard size.width > 1, size.height > 1 else { return nil }
        let pixelsWide = Int(size.width * scale)
        let pixelsHigh = Int(size.height * scale)
        guard pixelsWide > 0, pixelsHigh > 0,
              let rep = NSBitmapImageRep(
                bitmapDataPlanes: nil,
                pixelsWide: pixelsWide,
                pixelsHigh: pixelsHigh,
                bitsPerSample: 8,
                samplesPerPixel: 4,
                hasAlpha: true,
                isPlanar: false,
                colorSpaceName: .deviceRGB,
                bytesPerRow: 0,
                bitsPerPixel: 0
              )
        else { return nil }
        rep.size = size

        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        guard let gctx = NSGraphicsContext(bitmapImageRep: rep) else { return nil }
        NSGraphicsContext.current = gctx
        let ctx = gctx.cgContext

        let attributed = NSAttributedString(string: text, attributes: attributes)
        let textSize = attributed.size()
        let origin = NSPoint(x: (size.width - textSize.width) / 2.0,
                             y: (size.height - textSize.height) / 2.0)
        attributed.draw(at: origin)

        // Punch transparent bands through the glyphs. destinationOut keeps the
        // stripe geometry independent of glyph shape, exactly like the CSS
        // repeating-gradient clipped to text.
        ctx.saveGState()
        ctx.setBlendMode(.destinationOut)
        let stripe = max(1.0, pointSize * stripeRatio)
        let period = stripe * 1.92
        ctx.setFillColor(NSColor.black.cgColor)
        var y = origin.y
        while y < size.height {
            ctx.fill(CGRect(x: 0, y: y + stripe, width: size.width, height: period - stripe))
            y += period
        }
        ctx.restoreGState()

        // Tint the surviving stripes.
        ctx.saveGState()
        ctx.setBlendMode(.sourceIn)
        ctx.setFillColor(tint.cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))
        ctx.restoreGState()

        let image = NSImage(size: size)
        image.addRepresentation(rep)
        return image
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let scale = window?.backingScaleFactor ?? 2.0
        if cachedImage == nil || cachedScale != scale {
            cachedImage = stripedImage(scale: scale)
            cachedScale = scale
        }
        guard let image = cachedImage else { return }
        let rect = NSRect(origin: .zero, size: image.size)

        // Layered bloom, mirroring the web build's drop-shadow chain.
        for (blur, alpha) in [(3.0, 0.90), (14.0, 0.50), (40.0, 0.24)] {
            ctx.saveGState()
            ctx.setShadow(offset: .zero,
                          blur: CGFloat(blur),
                          color: glow.withAlphaComponent(CGFloat(alpha)).cgColor)
            image.draw(in: rect)
            ctx.restoreGState()
        }
        image.draw(in: rect)
    }
}

/// Aura's neuron mark — a retro-arcade cell.
///
/// Chunky low-poly soma, dendrites that branch out to square terminal nodes,
/// and a myelinated axon. Signal "spikes" are drawn as PIXELS (square, no
/// corner radius) and are driven along the real dendrite/axon paths by
/// CAKeyframeAnimations bound to those exact paths, so a spike is always on
/// its fibre rather than approximated with offsets. A scanline overlay and a
/// stepped soma pulse give it the CRT/arcade feel.
private final class AuraSigilView: NSView {
    private let cage = CALayer()

    init(diameter: CGFloat) {
        super.init(frame: NSRect(x: 0, y: 0, width: diameter, height: diameter))
        translatesAutoresizingMaskIntoConstraints = false
        wantsLayer = true
        build(diameter: diameter)
        NSLayoutConstraint.activate([
            widthAnchor.constraint(equalToConstant: diameter),
            heightAnchor.constraint(equalToConstant: diameter),
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    /// A square "pixel" node — the arcade vocabulary. Round dots read as an
    /// atom; squares read as a sprite.
    private func pixel(size: CGFloat, color: NSColor, at point: CGPoint) -> CALayer {
        let node = CALayer()
        node.bounds = CGRect(x: 0, y: 0, width: size, height: size)
        node.position = point
        node.backgroundColor = color.cgColor
        node.shadowColor = color.cgColor
        node.shadowOpacity = 0.95
        node.shadowRadius = size * 1.1
        node.shadowOffset = .zero
        return node
    }

    private func build(diameter: CGFloat) {
        guard let root = layer else { return }
        let center = CGPoint(x: diameter / 2, y: diameter / 2)
        let somaR = diameter * 0.135

        cage.frame = CGRect(x: 0, y: 0, width: diameter, height: diameter)
        root.addSublayer(cage)

        // ── Dendrites: branching fibres reaching up/out from the soma ──────
        // (angle, length, whether it forks)
        let dendrites: [(angle: CGFloat, reach: CGFloat, fork: Bool)] = [
            (128, 0.40, true),
            (168, 0.34, false),
            (96,  0.36, true),
            (212, 0.32, true),
            (58,  0.30, false),
        ]

        var spikePaths: [(path: CGPath, period: Double, color: NSColor)] = []

        for dendrite in dendrites {
            let radians = dendrite.angle * .pi / 180
            let reach = diameter * dendrite.reach
            // Step out from the soma edge in two segments so the fibre has a
            // hand-drawn kink rather than a straight ray.
            let root0 = CGPoint(x: center.x + cos(radians) * somaR,
                                y: center.y + sin(radians) * somaR)
            let mid = CGPoint(x: center.x + cos(radians - 0.16) * reach * 0.58,
                              y: center.y + sin(radians - 0.16) * reach * 0.58)
            let tip = CGPoint(x: center.x + cos(radians + 0.08) * reach,
                              y: center.y + sin(radians + 0.08) * reach)

            let fibre = CGMutablePath()
            fibre.move(to: root0)
            fibre.addLine(to: mid)
            fibre.addLine(to: tip)

            let branch = CAShapeLayer()
            branch.frame = cage.bounds
            branch.path = fibre
            branch.fillColor = nil
            branch.strokeColor = NSColor.auraCyan.withAlphaComponent(0.62).cgColor
            branch.lineWidth = 1.8
            branch.lineJoin = .miter
            branch.lineCap = .square
            branch.shadowColor = NSColor.auraCyan.cgColor
            branch.shadowOpacity = 0.7
            branch.shadowRadius = 4
            branch.shadowOffset = .zero
            cage.addSublayer(branch)

            cage.addSublayer(pixel(size: diameter * 0.030, color: .auraCyan, at: tip))

            if dendrite.fork {
                let forkTip = CGPoint(x: center.x + cos(radians + 0.34) * reach * 0.88,
                                      y: center.y + sin(radians + 0.34) * reach * 0.88)
                let twig = CGMutablePath()
                twig.move(to: mid)
                twig.addLine(to: forkTip)
                let twigLayer = CAShapeLayer()
                twigLayer.frame = cage.bounds
                twigLayer.path = twig
                twigLayer.fillColor = nil
                twigLayer.strokeColor = NSColor.auraCyan.withAlphaComponent(0.42).cgColor
                twigLayer.lineWidth = 1.3
                twigLayer.lineCap = .square
                cage.addSublayer(twigLayer)
                cage.addSublayer(pixel(size: diameter * 0.022, color: .auraCyan, at: forkTip))
            }

            // Spikes travel INWARD (tip -> soma), the direction a dendrite
            // actually carries signal.
            let inbound = CGMutablePath()
            inbound.move(to: tip)
            inbound.addLine(to: mid)
            inbound.addLine(to: root0)
            spikePaths.append((inbound, 2.2 + Double(dendrite.reach) * 2.4, .auraCyan))
        }

        // ── Axon: down-right, myelin-segmented, with terminal boutons ──────
        let axonStart = CGPoint(x: center.x + somaR * 0.72, y: center.y - somaR * 0.72)
        let axonKnee = CGPoint(x: center.x + diameter * 0.20, y: center.y - diameter * 0.26)
        let axonEnd = CGPoint(x: center.x + diameter * 0.40, y: center.y - diameter * 0.33)

        let axon = CGMutablePath()
        axon.move(to: axonStart)
        axon.addLine(to: axonKnee)
        axon.addLine(to: axonEnd)

        let axonLayer = CAShapeLayer()
        axonLayer.frame = cage.bounds
        axonLayer.path = axon
        axonLayer.fillColor = nil
        axonLayer.strokeColor = NSColor.auraViolet.withAlphaComponent(0.85).cgColor
        axonLayer.lineWidth = 2.6
        axonLayer.lineCap = .butt
        axonLayer.lineJoin = .miter
        // Myelin sheath, drawn as chunky dashes — reads as segmented armour.
        axonLayer.lineDashPattern = [6, 3]
        axonLayer.shadowColor = NSColor.auraViolet.cgColor
        axonLayer.shadowOpacity = 0.8
        axonLayer.shadowRadius = 5
        axonLayer.shadowOffset = .zero
        cage.addSublayer(axonLayer)

        // Terminal boutons: three square pads at the axon end.
        for offset in [CGPoint(x: 0, y: 0),
                       CGPoint(x: diameter * 0.045, y: diameter * 0.050),
                       CGPoint(x: diameter * 0.052, y: -diameter * 0.042)] {
            let pad = CGPoint(x: axonEnd.x + offset.x, y: axonEnd.y + offset.y)
            let stub = CGMutablePath()
            stub.move(to: axonEnd)
            stub.addLine(to: pad)
            let stubLayer = CAShapeLayer()
            stubLayer.frame = cage.bounds
            stubLayer.path = stub
            stubLayer.fillColor = nil
            stubLayer.strokeColor = NSColor.auraViolet.withAlphaComponent(0.6).cgColor
            stubLayer.lineWidth = 1.4
            stubLayer.lineCap = .square
            cage.addSublayer(stubLayer)
            cage.addSublayer(pixel(size: diameter * 0.028, color: .auraViolet, at: pad))
        }

        spikePaths.append((axon, 1.7, .auraViolet))

        // ── Travelling spikes ─────────────────────────────────────────────
        for (index, spike) in spikePaths.enumerated() {
            let dot = pixel(size: diameter * 0.034, color: .white, at: center)
            dot.shadowColor = spike.color.cgColor
            dot.shadowRadius = diameter * 0.05
            cage.addSublayer(dot)

            let travel = CAKeyframeAnimation(keyPath: "position")
            travel.path = spike.path
            travel.duration = spike.period
            travel.repeatCount = .infinity
            // Stepped, not paced: a spike should tick along its fibre like a
            // sprite on a grid rather than glide.
            travel.calculationMode = .discrete
            travel.timeOffset = spike.period * (0.13 * Double(index + 1))
            travel.isRemovedOnCompletion = false
            dot.add(travel, forKey: "spike")
        }

        // ── Soma: chunky hexagon, not a sphere ────────────────────────────
        let soma = CGMutablePath()
        for corner in 0..<6 {
            let a = (CGFloat(corner) * 60 - 90) * .pi / 180
            let point = CGPoint(x: center.x + cos(a) * somaR, y: center.y + sin(a) * somaR)
            if corner == 0 { soma.move(to: point) } else { soma.addLine(to: point) }
        }
        soma.closeSubpath()

        let somaLayer = CAShapeLayer()
        somaLayer.frame = cage.bounds
        somaLayer.path = soma
        somaLayer.fillColor = NSColor.auraViolet.cgColor
        somaLayer.strokeColor = NSColor.auraCyan.withAlphaComponent(0.9).cgColor
        somaLayer.lineWidth = 2.0
        somaLayer.lineJoin = .miter
        somaLayer.shadowColor = NSColor.auraCyan.cgColor
        somaLayer.shadowOpacity = 0.9
        somaLayer.shadowRadius = 13
        somaLayer.shadowOffset = .zero
        root.addSublayer(somaLayer)

        // Nucleus pixel.
        root.addSublayer(pixel(size: somaR * 0.52, color: NSColor.white.withAlphaComponent(0.92),
                               at: CGPoint(x: center.x, y: center.y + somaR * 0.06)))

        // Stepped "firing" pulse — discrete frames, like an arcade sprite
        // cycling, rather than a smooth breath.
        let fire = CAKeyframeAnimation(keyPath: "opacity")
        fire.values = [0.62, 1.0, 0.78, 1.0, 0.66]
        fire.keyTimes = [0, 0.18, 0.42, 0.63, 1.0]
        fire.calculationMode = .discrete
        fire.duration = 1.9
        fire.repeatCount = .infinity
        somaLayer.add(fire, forKey: "fire")

        // ── CRT scanlines over the whole mark ─────────────────────────────
        let scan = CAShapeLayer()
        scan.frame = cage.bounds
        let lines = CGMutablePath()
        var y: CGFloat = 0
        while y < diameter {
            lines.move(to: CGPoint(x: 0, y: y))
            lines.addLine(to: CGPoint(x: diameter, y: y))
            y += 3
        }
        scan.path = lines
        scan.strokeColor = NSColor.black.withAlphaComponent(0.22).cgColor
        scan.lineWidth = 1
        root.addSublayer(scan)
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
    private var isHovered = false
    private var trackingArea: NSTrackingArea?

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
        layer?.cornerRadius = 21
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

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let existing = trackingArea {
            removeTrackingArea(existing)
        }
        let options: NSTrackingArea.Options = [.mouseEnteredAndExited, .activeInActiveApp]
        let area = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    override func mouseEntered(with event: NSEvent) {
        isHovered = true
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.allowsImplicitAnimation = true
            updateAppearance()
        }, completionHandler: nil)
    }

    override func mouseExited(with event: NSEvent) {
        isHovered = false
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.allowsImplicitAnimation = true
            updateAppearance()
        }, completionHandler: nil)
    }

    private func updateAppearance() {
        let font = NSFont.systemFont(ofSize: 15, weight: .semibold)
        switch style {
        case .accent:
            let bgAlpha: CGFloat = isHovered ? 0.35 : 0.22
            let borderAlpha: CGFloat = isHovered ? 0.66 : 0.44
            layer?.backgroundColor = NSColor.auraViolet.withAlphaComponent(bgAlpha).cgColor
            layer?.borderColor = NSColor.auraViolet.withAlphaComponent(borderAlpha).cgColor
            layer?.shadowColor = NSColor.auraViolet.cgColor
            layer?.shadowOpacity = isHovered ? 0.42 : 0.0
            layer?.shadowRadius = isHovered ? 8 : 0
            layer?.shadowOffset = .zero
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: NSColor.white,
                ]
            )
        case .secondary:
            let bgAlpha: CGFloat = isHovered ? 0.14 : 0.04
            let borderAlpha: CGFloat = isHovered ? 0.22 : 0.10
            let textColor = isHovered ? NSColor.white : NSColor(calibratedWhite: 0.92, alpha: 1.0)
            layer?.backgroundColor = NSColor.white.withAlphaComponent(bgAlpha).cgColor
            layer?.borderColor = NSColor.white.withAlphaComponent(borderAlpha).cgColor
            layer?.shadowColor = NSColor.white.cgColor
            layer?.shadowOpacity = isHovered ? 0.15 : 0.0
            layer?.shadowRadius = isHovered ? 6 : 0
            layer?.shadowOffset = .zero
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: textColor,
                ]
            )
        case .subtle:
            let bgAlpha: CGFloat = isHovered ? 0.22 : 0.14
            let borderAlpha: CGFloat = isHovered ? 0.16 : 0.08
            layer?.backgroundColor = NSColor.black.withAlphaComponent(bgAlpha).cgColor
            layer?.borderColor = NSColor.white.withAlphaComponent(borderAlpha).cgColor
            attributedTitle = NSAttributedString(
                string: title,
                attributes: [
                    .font: font,
                    .foregroundColor: NSColor.auraTextMuted,
                ]
            )
        case .danger:
            let bgAlpha: CGFloat = isHovered ? 0.50 : 0.34
            let borderAlpha: CGFloat = isHovered ? 0.50 : 0.36
            layer?.backgroundColor = NSColor(calibratedRed: 0.46, green: 0.14, blue: 0.21, alpha: bgAlpha).cgColor
            layer?.borderColor = NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.67, alpha: borderAlpha).cgColor
            layer?.shadowColor = NSColor(calibratedRed: 1.0, green: 0.42, blue: 0.67, alpha: 1.0).cgColor
            layer?.shadowOpacity = isHovered ? 0.40 : 0.0
            layer?.shadowRadius = isHovered ? 8 : 0
            layer?.shadowOffset = .zero
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

private final class CircleCloseButton: NSButton {
    private var isHovered = false
    private var trackingArea: NSTrackingArea?

    init(target: AnyObject?, action: Selector) {
        super.init(frame: .zero)
        self.target = target
        self.action = action
        isBordered = false
        focusRingType = .none
        wantsLayer = true
        layer?.cornerRadius = 9
        layer?.borderWidth = 1.2
        updateAppearance()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var intrinsicContentSize: NSSize {
        NSSize(width: 18, height: 18)
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let existing = trackingArea {
            removeTrackingArea(existing)
        }
        let options: NSTrackingArea.Options = [.mouseEnteredAndExited, .activeInActiveApp]
        let area = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    override func mouseEntered(with event: NSEvent) {
        isHovered = true
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.15
            context.allowsImplicitAnimation = true
            updateAppearance()
        }, completionHandler: nil)
    }

    override func mouseExited(with event: NSEvent) {
        isHovered = false
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.15
            context.allowsImplicitAnimation = true
            updateAppearance()
        }, completionHandler: nil)
    }

    private func updateAppearance() {
        if isHovered {
            layer?.backgroundColor = NSColor(calibratedRed: 0.92, green: 0.26, blue: 0.35, alpha: 1.0).cgColor
            layer?.borderColor = NSColor(calibratedRed: 1.0, green: 0.45, blue: 0.55, alpha: 1.0).cgColor
            layer?.shadowColor = NSColor(calibratedRed: 1.0, green: 0.26, blue: 0.35, alpha: 0.6).cgColor
            layer?.shadowOpacity = 0.8
            layer?.shadowRadius = 4
            layer?.shadowOffset = .zero
            attributedTitle = NSAttributedString(
                string: "×",
                attributes: [
                    .font: NSFont.systemFont(ofSize: 12, weight: .bold),
                    .foregroundColor: NSColor(calibratedRed: 0.3, green: 0.0, blue: 0.05, alpha: 1.0),
                ]
            )
        } else {
            layer?.backgroundColor = NSColor(calibratedRed: 0.36, green: 0.11, blue: 0.15, alpha: 0.8).cgColor
            layer?.borderColor = NSColor(calibratedRed: 0.66, green: 0.22, blue: 0.28, alpha: 0.4).cgColor
            layer?.shadowOpacity = 0.0
            attributedTitle = NSAttributedString(
                string: "",
                attributes: [:]
            )
        }
    }
}

/// A borderless panel that is still allowed to hold keyboard focus.
///
/// AppKit decides `canBecomeKey` from the style mask, and a `.borderless`
/// window answers false no matter what else it is given. So the companion
/// chat — deliberately borderless, deliberately NOT `.nonactivatingPanel`
/// because it exists to be typed into — was ordered front by
/// `makeKeyAndOrderFront` and never became key. The composer rendered, took
/// clicks, and could not receive a single keystroke.
///
/// Reported live 2026-08-10: "i cant type in the mini chat".
///
/// Only the bubble stays non-keyable: it is a glyph you click, and taking
/// focus from whatever someone is working in is the one thing it must not do.
final class KeyablePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

/// A pan recognizer that only claims drags beginning in a strip along the top
/// edge of its view.
///
/// The companion transcript has to stay selectable, so the whole surface
/// cannot be a drag handle. The page marked its title bar with
/// `-webkit-app-region: drag`, which is an Electron property that WKWebView
/// does not implement — the intended handle was never live on any surface.
final class TopStripPanGestureRecognizer: NSPanGestureRecognizer {
    /// Height in points of the draggable strip from the top edge. Zero drags
    /// from anywhere, which is what the bubble wants: it is all glyph.
    var topStrip: CGFloat = 0

    override func mouseDown(with event: NSEvent) {
        if topStrip > 0, let view {
            let point = view.convert(event.locationInWindow, from: nil)
            let distanceFromTop = view.isFlipped ? point.y : view.bounds.height - point.y
            if distanceFromTop > topStrip {
                state = .failed
                return
            }
        }
        super.mouseDown(with: event)
    }
}

final class AuraLauncherDelegate: NSObject, NSApplicationDelegate,
    WKScriptMessageHandler, NSWindowDelegate, WKUIDelegate {
    private static let showPrimaryWindowNotification = Notification.Name(
        "com.aura.desktop.show-primary-window"
    )

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
    private var desktopWindow: NSWindow?
    private var desktopWebView: WKWebView?

    // ── the bubble ────────────────────────────────────────────────────
    //
    // When the desktop window closes, Aura is still running, and the bubble
    // is how she stays reachable without occupying the screen. It is a
    // non-activating floating panel: clicking it must not steal focus from
    // whatever the person is actually working in, because a companion that
    // pulls focus is an interruption even when it says nothing.
    private var bubblePanel: NSPanel?
    private var bubbleWebView: WKWebView?
    private var bubbleFrameObserver: Any?
    private var pendingBubbleMoveSequence: Int?
    private var companionPanel: NSPanel?
    private var companionWebView: WKWebView?
    private var overlayWindow: NSWindow?
    private var overlayDismissWork: DispatchWorkItem?
    // Screen-space anchors for a window drag in progress. Screen space, not
    // window space: the window moves WITH the cursor, so a translation measured
    // against the window nets to zero after the first step and the drag stalls.
    private var dragMouseAnchor: NSPoint?
    private var dragWindowAnchor: NSPoint?

    private var auraRoot: URL!
    private var launchScript: URL!
    private var pythonExecutable: URL!
    private var auraMainScript: URL!
    private var logFile: URL!
    private var lockDirectory: URL!
    private var bootMarkerFile: URL!
    private var bootBlockedFile: URL!
    private var terminalHandoffMarkerFile: URL!
    private var guiWindowMarkerFile: URL!
    private var spawnLockFile: URL!
    private var appInstanceLockFile: URL!
    private var nativeBridgeRequestDirectory: URL!
    private var nativeBridgeResponseDirectory: URL!
    private var launchProvenanceEnvironment: [String: String] = [:]
    private var appInstanceLockFD: Int32 = -1
    private let spawnedProcessesLock = NSLock()
    private var spawnedProcesses: [Process] = []

    private var pollTimer: Timer?
    private var nativeBridgeTimer: Timer?
    private var isPolling = false
    private var nativeBridgeProcessing = false
    private var launchInFlight = false
    private var lastSnapshot: BootSnapshot?
    private var bundledSemver: String = ""
    private var bundledVersionLabel: String = ""
    private var forcedRelaunchAttempted = false
    private var autoDesktopOpenTriggered = false
    private var spawnedFreshRuntime = false
    private var explicitStopInProgress = false
    private let launchedAt = Date()
    private let staleMarkerWithoutRuntimeWindow: TimeInterval = 8.0
    private let terminalHandoffWindow: TimeInterval = 75.0
    private let guiWindowLaunchWindow: TimeInterval = 25.0

    /// Install the standard menu bar.
    ///
    /// LIVE DEFECT, 2026-08-03. Bryan reported that copy, paste and select-all
    /// do nothing in Aura's window. The app never built a menu bar, and on
    /// macOS ⌘C/⌘V/⌘X/⌘A reach a WKWebView through the Edit menu's key
    /// equivalents — an app with no Edit menu has no clipboard shortcuts at
    /// all. Nothing in the web content could have fixed it; the responder
    /// chain was never asked.
    ///
    /// These are the standard selectors, so they act on whatever has focus
    /// (the web view, a text field) exactly as every other Mac app behaves.
    private func installMenuBar() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()
        let appName = ProcessInfo.processInfo.processName
        appMenu.addItem(
            withTitle: "Hide \(appName)",
            action: #selector(NSApplication.hide(_:)),
            keyEquivalent: "h",
        )
        let hideOthers = appMenu.addItem(
            withTitle: "Hide Others",
            action: #selector(NSApplication.hideOtherApplications(_:)),
            keyEquivalent: "h",
        )
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(
            withTitle: "Show All",
            action: #selector(NSApplication.unhideAllApplications(_:)),
            keyEquivalent: "",
        )
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(
            withTitle: "Quit \(appName)",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q",
        )
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let editMenuItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(
            withTitle: "Undo",
            action: Selector(("undo:")),
            keyEquivalent: "z",
        )
        let redo = editMenu.addItem(
            withTitle: "Redo",
            action: Selector(("redo:")),
            keyEquivalent: "z",
        )
        redo.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(NSMenuItem.separator())
        editMenu.addItem(
            withTitle: "Cut",
            action: #selector(NSText.cut(_:)),
            keyEquivalent: "x",
        )
        editMenu.addItem(
            withTitle: "Copy",
            action: #selector(NSText.copy(_:)),
            keyEquivalent: "c",
        )
        editMenu.addItem(
            withTitle: "Paste",
            action: #selector(NSText.paste(_:)),
            keyEquivalent: "v",
        )
        let pasteMatch = editMenu.addItem(
            withTitle: "Paste and Match Style",
            action: #selector(NSTextView.pasteAsPlainText(_:)),
            keyEquivalent: "v",
        )
        pasteMatch.keyEquivalentModifierMask = [.command, .option, .shift]
        editMenu.addItem(
            withTitle: "Delete",
            action: #selector(NSText.delete(_:)),
            keyEquivalent: "",
        )
        editMenu.addItem(
            withTitle: "Select All",
            action: #selector(NSText.selectAll(_:)),
            keyEquivalent: "a",
        )
        editMenuItem.submenu = editMenu
        mainMenu.addItem(editMenuItem)

        let windowMenuItem = NSMenuItem()
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(
            withTitle: "Minimize",
            action: #selector(NSWindow.performMiniaturize(_:)),
            keyEquivalent: "m",
        )
        windowMenu.addItem(
            withTitle: "Zoom",
            action: #selector(NSWindow.performZoom(_:)),
            keyEquivalent: "",
        )
        windowMenuItem.submenu = windowMenu
        mainMenu.addItem(windowMenuItem)

        NSApp.mainMenu = mainMenu
        NSApp.windowsMenu = windowMenu
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        installMenuBar()
        do {
            try configurePaths()
        } catch {
            showFatalError(
                title: "Aura Launcher Error",
                detail: error.localizedDescription,
            )
            return
        }

        guard claimAppInstanceLock() else {
            return
        }

        DistributedNotificationCenter.default().addObserver(
            self,
            selector: #selector(handleShowPrimaryWindowNotification(_:)),
            name: Self.showPrimaryWindowNotification,
            object: nil,
        )

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
        nativeBridgeTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.processNativeBridgeRequests()
        }
    }

    @discardableResult
    private func frontPrimaryWindow() -> Bool {
        guard let target = desktopWindow ?? window else {
            return false
        }
        target.makeKeyAndOrderFront(nil)
        target.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [])
        NSApp.requestUserAttention(.informationalRequest)
        return true
    }

    @objc private func handleShowPrimaryWindowNotification(_ notification: Notification) {
        if let targetPID = notification.userInfo?["target_pid"] as? NSNumber,
           targetPID.int32Value != ProcessInfo.processInfo.processIdentifier {
            return
        }
        frontPrimaryWindow()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        frontPrimaryWindow()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationWillTerminate(_ notification: Notification) {
        // A direct launcher child belongs to this app lifecycle. Command-Q,
        // AppleScript quit, logout, and explicit stop must all reap it; an
        // attached pre-existing runtime is absent from spawnedProcesses.
        terminateSpawnedProcesses()
        DistributedNotificationCenter.default().removeObserver(self)
        releaseAppInstanceLock()
    }

    private func configurePaths() throws {
        guard let resourcesURL = Bundle.main.resourceURL else {
            throw NSError(domain: "AuraLauncher", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Bundle resources are missing.",
            ])
        }

        let rootFallback = resourcesURL.appendingPathComponent("aura-root-path")
        guard let rootText = try? String(contentsOf: rootFallback, encoding: .utf8) else {
            throw NSError(domain: "AuraLauncher", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Aura root path is missing. Rebuild the launcher from the repo.",
            ])
        }
        let trimmedRoot = rootText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoot.isEmpty else {
            throw NSError(domain: "AuraLauncher", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Aura root path is empty. Rebuild the launcher from the repo.",
            ])
        }
        auraRoot = URL(fileURLWithPath: trimmedRoot, isDirectory: true)
            .resolvingSymlinksInPath()
            .standardizedFileURL

        let provenanceURL = resourcesURL.appendingPathComponent("aura-launch-provenance.json")
        guard let provenanceData = try? Data(contentsOf: provenanceURL),
              let provenance = try? JSONSerialization.jsonObject(with: provenanceData) as? [String: Any] else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura launch provenance is missing or invalid. Rebuild the installed app.",
            ])
        }
        func requiredProvenanceString(_ key: String) throws -> String {
            let value = String(describing: provenance[key] ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty else {
                throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Aura launch provenance is missing \(key). Rebuild the installed app.",
                ])
            }
            return value
        }
        let schema = try requiredProvenanceString("schema")
        guard schema == "aura.launch_provenance.v1" else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura launch provenance uses an unsupported schema.",
            ])
        }
        let shellAssetsSHA256 = try requiredProvenanceString("shell_assets_sha256")
        let lowercaseHex = CharacterSet(charactersIn: "0123456789abcdef")
        guard shellAssetsSHA256.count == 64,
              shellAssetsSHA256.unicodeScalars.allSatisfy({ lowercaseHex.contains($0) }) else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura launch provenance has an invalid runtime shell digest.",
            ])
        }
        let manifestRoot = URL(
            fileURLWithPath: try requiredProvenanceString("source_root"),
            isDirectory: true
        ).resolvingSymlinksInPath().standardizedFileURL
        guard manifestRoot.path == auraRoot.path else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura.app source root does not match its signed launch manifest.",
            ])
        }
        guard let appExecutable = Bundle.main.executableURL?.resolvingSymlinksInPath() else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura.app could not resolve its signed launcher executable.",
            ])
        }
        let bundleIdentifier = Bundle.main.bundleIdentifier ?? ""
        guard bundleIdentifier == "com.aura.desktop" else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura.app bundle identity does not match com.aura.desktop.",
            ])
        }
        launchProvenanceEnvironment = [
            "AURA_LAUNCH_MANIFEST_PATH": provenanceURL.path,
            "AURA_LAUNCH_APP_EXECUTABLE": appExecutable.path,
            "AURA_LAUNCH_EXPECTED_ROOT": manifestRoot.path,
            "AURA_LAUNCH_EXPECTED_COMMIT": try requiredProvenanceString("commit_sha"),
            "AURA_LAUNCH_EXPECTED_BRANCH": try requiredProvenanceString("branch"),
            "AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256": try requiredProvenanceString("workspace_state_sha256"),
            "AURA_LAUNCH_BUNDLE_ID": bundleIdentifier,
        ]
        let modelsPathFile = resourcesURL.appendingPathComponent("aura-models-path")
        guard let modelsPath = try? String(contentsOf: modelsPathFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !modelsPath.isEmpty else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura's signed model inventory path is missing. Rebuild the installed app.",
            ])
        }
        let modelsURL = URL(fileURLWithPath: modelsPath, isDirectory: true).standardizedFileURL
        var isModelsDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: modelsURL.path, isDirectory: &isModelsDirectory),
              isModelsDirectory.boolValue,
              fileManager.isReadableFile(atPath: modelsURL.path) else {
            throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Aura's signed model inventory is unavailable. Restore it or rebuild the installed app.",
            ])
        }
        launchProvenanceEnvironment["AURA_MODELS_DIR"] = modelsURL.path
        let fusedModelPathFile = resourcesURL.appendingPathComponent("aura-fused-model-path")
        if let fusedModelPath = try? String(contentsOf: fusedModelPathFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !fusedModelPath.isEmpty {
            let fusedModelURL = URL(
                fileURLWithPath: fusedModelPath,
                isDirectory: true
            ).standardizedFileURL
            var isFusedModelDirectory: ObjCBool = false
            if fileManager.fileExists(
                atPath: fusedModelURL.path,
                isDirectory: &isFusedModelDirectory
            ), isFusedModelDirectory.boolValue,
               fileManager.isReadableFile(atPath: fusedModelURL.path) {
                launchProvenanceEnvironment["AURA_FUSED_MODEL_ROOT"] = fusedModelURL.path
            }
        }
        let envPathFile = resourcesURL.appendingPathComponent("aura-env-path")
        if let envPath = try? String(contentsOf: envPathFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !envPath.isEmpty {
            let envURL = URL(fileURLWithPath: envPath).standardizedFileURL
            guard fileManager.isReadableFile(atPath: envURL.path) else {
                throw NSError(domain: "AuraLauncher", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Aura's signed runtime configuration is unavailable. Restore it or rebuild the installed app.",
                ])
            }
            launchProvenanceEnvironment["AURA_ENV_FILE"] = envURL.path
        }

        launchScript = auraRoot.appendingPathComponent("launch_aura.sh")
        auraMainScript = auraRoot.appendingPathComponent("aura_main.py")
        pythonExecutable = try resolvePythonExecutable(resourcesURL: resourcesURL)

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
        guiWindowMarkerFile = lockDirectory.appendingPathComponent("desktop-gui-window.marker")
        bootBlockedFile = auraHome.appendingPathComponent("run", isDirectory: true)
            .appendingPathComponent("boot_blocked.json")
        spawnLockFile = lockDirectory.appendingPathComponent("desktop-app-launch.lock")
        appInstanceLockFile = lockDirectory.appendingPathComponent("desktop-app-instance.lock")
        let nativeBridgeDirectory = auraHome.appendingPathComponent("native_bridge", isDirectory: true)
        nativeBridgeRequestDirectory = nativeBridgeDirectory.appendingPathComponent("requests", isDirectory: true)
        nativeBridgeResponseDirectory = nativeBridgeDirectory.appendingPathComponent("responses", isDirectory: true)

        try fileManager.createDirectory(at: logDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: lockDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: nativeBridgeRequestDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: nativeBridgeResponseDirectory, withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: logFile.path) {
            fileManager.createFile(atPath: logFile.path, contents: Data())
        }
    }

    private func claimAppInstanceLock() -> Bool {
        let fd = open(appInstanceLockFile.path, O_CREAT | O_RDWR, 0o644)
        guard fd != -1 else {
            return true
        }

        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            close(fd)
            activateExistingLauncherInstance()
            NSApp.terminate(nil)
            return false
        }

        appInstanceLockFD = fd
        ftruncate(fd, 0)
        let payload = "\(getpid())\n"
        _ = payload.withCString { pointer in
            write(fd, pointer, strlen(pointer))
        }
        fsync(fd)
        return true
    }

    private func releaseAppInstanceLock() {
        if appInstanceLockFD != -1 {
            flock(appInstanceLockFD, LOCK_UN)
            close(appInstanceLockFD)
            appInstanceLockFD = -1
        }
    }

    private func activateExistingLauncherInstance() {
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let bundleID = Bundle.main.bundleIdentifier ?? "com.aura.desktop"
        let candidates = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
            .filter { $0.processIdentifier != currentPID }
        if let existing = candidates.first {
            existing.activate(options: [.activateAllWindows])
            DistributedNotificationCenter.default().postNotificationName(
                Self.showPrimaryWindowNotification,
                object: nil,
                userInfo: ["target_pid": existing.processIdentifier],
                deliverImmediately: true,
            )
        }
    }

    private func processNativeBridgeRequests() {
        if nativeBridgeProcessing {
            return
        }
        nativeBridgeProcessing = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            defer {
                DispatchQueue.main.async {
                    self?.nativeBridgeProcessing = false
                }
            }
            guard let self,
                  let requestDirectory = self.nativeBridgeRequestDirectory,
                  let responseDirectory = self.nativeBridgeResponseDirectory else {
                return
            }
            guard let files = try? self.fileManager.contentsOfDirectory(
                at: requestDirectory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else {
                return
            }
            for requestURL in files where requestURL.pathExtension == "json" {
                self.handleNativeBridgeRequest(
                    requestURL: requestURL,
                    responseDirectory: responseDirectory
                )
            }
        }
    }

    private func handleNativeBridgeRequest(requestURL: URL, responseDirectory: URL) {
        let requestID = requestURL.deletingPathExtension().lastPathComponent
        guard !requestID.isEmpty else {
            try? fileManager.removeItem(at: requestURL)
            return
        }
        var response: [String: Any]
        var status: Int32 = 0
        if let data = try? Data(contentsOf: requestURL),
           let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let result = evaluateNativeDesktopBridge(payload: payload)
            response = result.0
            status = result.1
        } else {
            response = ["ok": false, "error": "invalid_bridge_payload"]
            status = 2
        }
        response["handled_by"] = "resident_aura_launcher"
        response["returncode"] = Int(status)
        let responseURL = responseDirectory.appendingPathComponent("\(requestID).json")
        let tmpURL = responseDirectory.appendingPathComponent(".\(requestID).json.tmp")
        if let data = try? JSONSerialization.data(withJSONObject: response, options: []) {
            do {
                try data.write(to: tmpURL, options: .atomic)
                if fileManager.fileExists(atPath: responseURL.path) {
                    try? fileManager.removeItem(at: responseURL)
                }
                try fileManager.moveItem(at: tmpURL, to: responseURL)
            } catch {
                try? fileManager.removeItem(at: tmpURL)
            }
        }
        try? fileManager.removeItem(at: requestURL)
    }

    private func buildWindow() {
        let frame = NSRect(x: 0, y: 0, width: 860, height: 500)
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
        window.standardWindowButton(.closeButton)?.isHidden = true
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true

        let contentView = LauncherBackgroundView(frame: frame)
        window.contentView = contentView

        let contentCard = NSVisualEffectView()
        contentCard.translatesAutoresizingMaskIntoConstraints = false
        contentCard.state = .active
        contentCard.material = .hudWindow
        contentCard.blendingMode = .behindWindow
        contentCard.wantsLayer = true
        contentCard.layer?.cornerRadius = 32
        contentCard.layer?.borderColor = NSColor.auraPanelBorder.cgColor
        contentCard.layer?.borderWidth = 1.2
        contentCard.layer?.shadowColor = NSColor.black.cgColor
        contentCard.layer?.shadowOpacity = 0.30
        contentCard.layer?.shadowRadius = 36
        contentCard.layer?.shadowOffset = .zero
        contentView.addSubview(contentCard)

        let closeButton = CircleCloseButton(target: self, action: #selector(closeLauncher))
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        contentCard.addSubview(closeButton)

        let heroPanel = NSView()
        heroPanel.translatesAutoresizingMaskIntoConstraints = false
        heroPanel.wantsLayer = true
        heroPanel.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.18).cgColor
        heroPanel.layer?.cornerRadius = 28
        heroPanel.layer?.borderColor = NSColor.white.withAlphaComponent(0.06).cgColor
        heroPanel.layer?.borderWidth = 1
        contentCard.addSubview(heroPanel)

        let actionTray = NSView()
        actionTray.translatesAutoresizingMaskIntoConstraints = false
        actionTray.wantsLayer = true
        actionTray.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.20).cgColor
        actionTray.layer?.cornerRadius = 22
        actionTray.layer?.borderColor = NSColor.white.withAlphaComponent(0.06).cgColor
        actionTray.layer?.borderWidth = 1
        contentCard.addSubview(actionTray)

        // AURA LAUNCHER — the brand lockup, scanline-cut to match the web splash.
        let eyebrowLabel = ScanlineWordmarkView(
            text: "AURA LUNA",
            tint: NSColor(calibratedRed: 0.79, green: 0.55, blue: 1.0, alpha: 1.0),
            glow: .auraViolet,
            pointSize: 26
        )

        let headerChips = NSStackView()
        headerChips.translatesAutoresizingMaskIntoConstraints = false
        headerChips.orientation = .horizontal
        headerChips.spacing = 8
        headerChips.alignment = .centerY
        let versionChip = LauncherChipLabel(
            bundledVersionLabel.isEmpty ? "Live Workspace" : bundledVersionLabel,
            tone: .violet
        )
        let routeChip = LauncherChipLabel("Local Boot Monitor", tone: .cyan)
        let portChip = LauncherChipLabel("127.0.0.1:8000", tone: .neutral)
        headerChips.addArrangedSubview(versionChip)
        headerChips.addArrangedSubview(routeChip)
        headerChips.addArrangedSubview(portChip)

        let orbHalo = NSView()
        orbHalo.translatesAutoresizingMaskIntoConstraints = false
        orbHalo.wantsLayer = true
        orbHalo.layer?.backgroundColor = NSColor.auraViolet.withAlphaComponent(0.24).cgColor
        orbHalo.layer?.cornerRadius = 76
        orbHalo.layer?.shadowColor = NSColor.auraViolet.cgColor
        orbHalo.layer?.shadowOpacity = 0.80
        orbHalo.layer?.shadowRadius = 60
        orbHalo.layer?.shadowOffset = .zero

        let pulse = CABasicAnimation(keyPath: "opacity")
        pulse.fromValue = 0.5
        pulse.toValue = 0.95
        pulse.duration = 2.8
        pulse.autoreverses = true
        pulse.repeatCount = .infinity
        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        orbHalo.layer?.add(pulse, forKey: "pulseOpacity")

        let pulseScale = CABasicAnimation(keyPath: "transform.scale")
        pulseScale.fromValue = 0.96
        pulseScale.toValue = 1.04
        pulseScale.duration = 2.8
        pulseScale.autoreverses = true
        pulseScale.repeatCount = .infinity
        pulseScale.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        orbHalo.layer?.add(pulseScale, forKey: "pulseScale")

        heroPanel.addSubview(orbHalo)

        let iconPlate = NSView()
        iconPlate.translatesAutoresizingMaskIntoConstraints = false
        iconPlate.wantsLayer = true
        iconPlate.layer?.backgroundColor = NSColor(calibratedRed: 0.06, green: 0.05, blue: 0.11, alpha: 0.72).cgColor
        iconPlate.layer?.cornerRadius = 30
        iconPlate.layer?.borderColor = NSColor.auraViolet.withAlphaComponent(0.34).cgColor
        iconPlate.layer?.borderWidth = 1
        iconPlate.layer?.shadowColor = NSColor.auraBlue.cgColor
        iconPlate.layer?.shadowOpacity = 0.32
        iconPlate.layer?.shadowRadius = 18
        iconPlate.layer?.shadowOffset = .zero
        heroPanel.addSubview(iconPlate)

        // Aura's own mark rather than the generic app icon: a neuron whose
        // spikes ride real dendrite and axon paths.
        let iconView = AuraSigilView(diameter: 92)
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

        let phaseSupportLabel = NSTextField(labelWithString: "Runtime handoff, health polling, and GUI wake coordination")
        phaseSupportLabel.translatesAutoresizingMaskIntoConstraints = false
        phaseSupportLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        phaseSupportLabel.textColor = NSColor.auraTextMuted.withAlphaComponent(0.95)
        phaseSupportLabel.lineBreakMode = .byTruncatingTail

        let summaryCard = NSView()
        summaryCard.translatesAutoresizingMaskIntoConstraints = false
        summaryCard.wantsLayer = true
        summaryCard.layer?.backgroundColor = NSColor.white.withAlphaComponent(0.04).cgColor
        summaryCard.layer?.cornerRadius = 22
        summaryCard.layer?.borderColor = NSColor.white.withAlphaComponent(0.08).cgColor
        summaryCard.layer?.borderWidth = 1
        heroPanel.addSubview(summaryCard)

        let summaryEyebrow = NSTextField(labelWithString: "BOOT HEALTH")
        summaryEyebrow.translatesAutoresizingMaskIntoConstraints = false
        summaryEyebrow.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
        summaryEyebrow.textColor = NSColor.auraCyan.withAlphaComponent(0.84)

        footerLabel = NSTextField(wrappingLabelWithString: "You can keep this window open while Aura boots.")
        footerLabel.translatesAutoresizingMaskIntoConstraints = false
        footerLabel.font = NSFont.systemFont(ofSize: 14, weight: .regular)
        footerLabel.textColor = NSColor.auraTextMuted
        footerLabel.maximumNumberOfLines = 3
        footerLabel.lineBreakMode = .byWordWrapping

        progressIndicator = GradientProgressBar()
        progressIndicator.translatesAutoresizingMaskIntoConstraints = false
        progressIndicator.progress = 6

        let progressBelowBadge = progressIndicator.topAnchor.constraint(greaterThanOrEqualTo: phaseBadge.bottomAnchor, constant: 24)
        progressBelowBadge.priority = .defaultHigh
        let progressBelowIcon = progressIndicator.topAnchor.constraint(greaterThanOrEqualTo: iconPlate.bottomAnchor, constant: 28)
        let progressBelowSummary = progressIndicator.topAnchor.constraint(greaterThanOrEqualTo: summaryCard.bottomAnchor, constant: 24)

        progressValueLabel = NSTextField(labelWithString: "6%")
        progressValueLabel.translatesAutoresizingMaskIntoConstraints = false
        progressValueLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 38, weight: .bold)
        progressValueLabel.textColor = NSColor.white

        let summaryCaption = NSTextField(labelWithString: "Boot API listening on the local desktop lane")
        summaryCaption.translatesAutoresizingMaskIntoConstraints = false
        summaryCaption.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        summaryCaption.textColor = NSColor.auraTextMuted
        summaryCaption.maximumNumberOfLines = 3
        summaryCaption.lineBreakMode = .byWordWrapping

        openLogsButton = CapsuleButton(title: "Open Logs", style: .secondary, target: self, action: #selector(openLogs))
        openLogsButton.translatesAutoresizingMaskIntoConstraints = false

        openDesktopButton = CapsuleButton(title: "Open Aura", style: .accent, target: self, action: #selector(openDesktopWindow))
        openDesktopButton.translatesAutoresizingMaskIntoConstraints = false

        openBrowserButton = CapsuleButton(title: "Open Browser", style: .secondary, target: self, action: #selector(openBrowser))
        openBrowserButton.translatesAutoresizingMaskIntoConstraints = false

        forceStopButton = CapsuleButton(title: "Force Stop", style: .danger, target: self, action: #selector(forceStopAura))
        forceStopButton.translatesAutoresizingMaskIntoConstraints = false

        heroPanel.addSubview(eyebrowLabel)
        heroPanel.addSubview(headerChips)
        heroPanel.addSubview(titleLabel)
        heroPanel.addSubview(detailLabel)
        heroPanel.addSubview(phaseBadge)
        heroPanel.addSubview(phaseSupportLabel)
        heroPanel.addSubview(progressIndicator)
        heroPanel.addSubview(footerLabel)
        summaryCard.addSubview(summaryEyebrow)
        summaryCard.addSubview(progressValueLabel)
        summaryCard.addSubview(summaryCaption)
        actionTray.addSubview(openLogsButton)
        actionTray.addSubview(openDesktopButton)
        actionTray.addSubview(openBrowserButton)
        actionTray.addSubview(forceStopButton)

        NSLayoutConstraint.activate([
            contentCard.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 22),
            contentCard.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -22),
            contentCard.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 22),
            contentCard.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -22),

            closeButton.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 18),
            closeButton.topAnchor.constraint(equalTo: contentCard.topAnchor, constant: 18),

            heroPanel.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 24),
            heroPanel.trailingAnchor.constraint(equalTo: contentCard.trailingAnchor, constant: -24),
            heroPanel.topAnchor.constraint(equalTo: contentCard.topAnchor, constant: 24),

            actionTray.leadingAnchor.constraint(equalTo: contentCard.leadingAnchor, constant: 24),
            actionTray.trailingAnchor.constraint(equalTo: contentCard.trailingAnchor, constant: -24),
            actionTray.bottomAnchor.constraint(equalTo: contentCard.bottomAnchor, constant: -24),
            actionTray.heightAnchor.constraint(equalToConstant: 82),

            heroPanel.bottomAnchor.constraint(equalTo: actionTray.topAnchor, constant: -18),

            eyebrowLabel.leadingAnchor.constraint(equalTo: heroPanel.leadingAnchor, constant: 32),
            eyebrowLabel.topAnchor.constraint(equalTo: heroPanel.topAnchor, constant: 28),

            headerChips.trailingAnchor.constraint(equalTo: heroPanel.trailingAnchor, constant: -32),
            headerChips.centerYAnchor.constraint(equalTo: eyebrowLabel.centerYAnchor),

            orbHalo.leadingAnchor.constraint(equalTo: heroPanel.leadingAnchor, constant: 24),
            orbHalo.topAnchor.constraint(equalTo: eyebrowLabel.bottomAnchor, constant: 14),
            orbHalo.widthAnchor.constraint(equalToConstant: 152),
            orbHalo.heightAnchor.constraint(equalToConstant: 152),

            iconPlate.leadingAnchor.constraint(equalTo: heroPanel.leadingAnchor, constant: 36),
            iconPlate.topAnchor.constraint(equalTo: eyebrowLabel.bottomAnchor, constant: 26),
            iconPlate.widthAnchor.constraint(equalToConstant: 116),
            iconPlate.heightAnchor.constraint(equalToConstant: 116),

            // The sigil carries its own 92pt intrinsic size; centre it in the
            // 116pt plate rather than pinning edges and fighting that.
            iconView.centerXAnchor.constraint(equalTo: iconPlate.centerXAnchor),
            iconView.centerYAnchor.constraint(equalTo: iconPlate.centerYAnchor),

            summaryCard.trailingAnchor.constraint(equalTo: heroPanel.trailingAnchor, constant: -32),
            summaryCard.topAnchor.constraint(equalTo: eyebrowLabel.bottomAnchor, constant: 18),
            summaryCard.widthAnchor.constraint(equalToConstant: 218),
            summaryCard.heightAnchor.constraint(equalToConstant: 148),

            summaryEyebrow.leadingAnchor.constraint(equalTo: summaryCard.leadingAnchor, constant: 18),
            summaryEyebrow.topAnchor.constraint(equalTo: summaryCard.topAnchor, constant: 18),

            progressValueLabel.leadingAnchor.constraint(equalTo: summaryEyebrow.leadingAnchor),
            progressValueLabel.topAnchor.constraint(equalTo: summaryEyebrow.bottomAnchor, constant: 10),

            summaryCaption.leadingAnchor.constraint(equalTo: summaryEyebrow.leadingAnchor),
            summaryCaption.trailingAnchor.constraint(equalTo: summaryCard.trailingAnchor, constant: -18),
            summaryCaption.topAnchor.constraint(equalTo: progressValueLabel.bottomAnchor, constant: 8),

            titleLabel.leadingAnchor.constraint(equalTo: iconPlate.trailingAnchor, constant: 28),
            titleLabel.trailingAnchor.constraint(equalTo: summaryCard.leadingAnchor, constant: -24),
            titleLabel.topAnchor.constraint(equalTo: iconPlate.topAnchor, constant: 2),

            detailLabel.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            detailLabel.trailingAnchor.constraint(equalTo: titleLabel.trailingAnchor),
            detailLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 8),

            phaseBadge.leadingAnchor.constraint(equalTo: titleLabel.leadingAnchor),
            phaseBadge.topAnchor.constraint(equalTo: detailLabel.bottomAnchor, constant: 12),
            phaseBadge.heightAnchor.constraint(equalToConstant: 28),
            phaseBadge.widthAnchor.constraint(greaterThanOrEqualToConstant: 164),

            phaseSupportLabel.leadingAnchor.constraint(equalTo: phaseBadge.trailingAnchor, constant: 12),
            phaseSupportLabel.centerYAnchor.constraint(equalTo: phaseBadge.centerYAnchor),
            phaseSupportLabel.trailingAnchor.constraint(lessThanOrEqualTo: summaryCard.leadingAnchor, constant: -18),

            progressIndicator.leadingAnchor.constraint(equalTo: heroPanel.leadingAnchor, constant: 32),
            progressIndicator.trailingAnchor.constraint(equalTo: heroPanel.trailingAnchor, constant: -32),
            progressBelowBadge,
            progressBelowIcon,
            progressBelowSummary,
            progressIndicator.heightAnchor.constraint(equalToConstant: 8),

            footerLabel.leadingAnchor.constraint(equalTo: heroPanel.leadingAnchor, constant: 32),
            footerLabel.trailingAnchor.constraint(equalTo: heroPanel.trailingAnchor, constant: -32),
            footerLabel.topAnchor.constraint(equalTo: progressIndicator.bottomAnchor, constant: 18),
            footerLabel.bottomAnchor.constraint(lessThanOrEqualTo: heroPanel.bottomAnchor, constant: -28),

            openLogsButton.leadingAnchor.constraint(equalTo: actionTray.leadingAnchor, constant: 20),
            openLogsButton.centerYAnchor.constraint(equalTo: actionTray.centerYAnchor),

            openDesktopButton.leadingAnchor.constraint(equalTo: openLogsButton.trailingAnchor, constant: 10),
            openDesktopButton.centerYAnchor.constraint(equalTo: actionTray.centerYAnchor),

            openBrowserButton.leadingAnchor.constraint(equalTo: openDesktopButton.trailingAnchor, constant: 10),
            openBrowserButton.centerYAnchor.constraint(equalTo: actionTray.centerYAnchor),

            forceStopButton.trailingAnchor.constraint(equalTo: actionTray.trailingAnchor, constant: -20),
            forceStopButton.centerYAnchor.constraint(equalTo: actionTray.centerYAnchor),
        ])
    }

    private func resolvePythonExecutable(resourcesURL: URL) throws -> URL {
        let runtimePathFile = resourcesURL.appendingPathComponent("aura-python-path")
        guard let runtimePath = try? String(contentsOf: runtimePathFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !runtimePath.isEmpty else {
            throw NSError(domain: "AuraLauncher", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Aura.app has no signed repository Python path. Rebuild the installed app.",
            ])
        }
        let candidate = URL(fileURLWithPath: runtimePath)
            .standardizedFileURL
        guard fileManager.isExecutableFile(atPath: candidate.path) else {
            throw NSError(domain: "AuraLauncher", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Aura's signed repository Python runtime is unavailable. Restore the repo environment or rebuild the installed app.",
            ])
        }
        return candidate
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
        // A normal app launch is always the full runtime.  Do not inherit a
        // stale recovery-only flag from launchd, Terminal, or a previous
        // diagnostic session.
        env.removeValue(forKey: "AURA_SAFE_BOOT_DESKTOP")
        env.removeValue(forKey: "AURA_AUTO_LISTEN")
        // Same rule for diagnostic reply-repair switches. A normal user launch
        // should recover through one RAM-gated same-worker Cortex retry instead
        // of inheriting a stale "disable repair" flag and failing closed.
        env.removeValue(forKey: "AURA_DESKTOP_ALLOW_SECONDARY_MODEL_REPAIR")
        env["AURA_LOCAL_BACKEND"] = "mlx"
        env["AURA_LAUNCHED_FROM_APP"] = "1"
        env["AURA_NATIVE_BRIDGE_PID"] = String(ProcessInfo.processInfo.processIdentifier)
        env["AURA_DESKTOP_RESOURCE_GUARD"] = "1"
        env["AURA_ENABLE_BACKGROUND_COGNITION"] = "1"
        env["AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM"] = "1"
        // The installed desktop is Aura's sovereign local operator profile.
        // Promotion still has to pass the existing quarantine, validation,
        // holdout, and rollback gates.
        env["AURA_ALLOW_RUNTIME_SELF_MODIFICATION"] = "1"
        env["AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION"] = "1"
        env["AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION"] = "1"
        // Resident model assets are source-bound and local. Keep desktop boot
        // deterministic and free of implicit Hugging Face network traffic.
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["AURA_BACKGROUND_BOOT_GRACE_S"] = "60"
        env["AURA_EAGER_LOCAL_SENSORY_BOOT"] = "1"
        env["AURA_EAGER_CORTEX_WARMUP"] = "0"
        env["AURA_DEFERRED_CORTEX_PREWARM"] = "1"
        env["AURA_DESKTOP_METAL_CACHE_RATIO"] = "0.16"
        env["AURA_DESKTOP_METAL_CACHE_CAP_GB"] = "10"
        env["AURA_DESKTOP_MLX_MEMORY_RATIO"] = "0.54"
        env["AURA_DESKTOP_MLX_MEMORY_CAP_GB"] = "34"
        env["AURA_DESKTOP_MLX_MEMORY_FLOOR_GB"] = "18"
        env["AURA_DESKTOP_PROCESS_RSS_RATIO"] = "0.62"
        env["AURA_DESKTOP_PROCESS_RSS_CAP_GB"] = "40"
        env["AURA_DESKTOP_PROCESS_RSS_FLOOR_GB"] = "24"
        env["AURA_PROCESS_RSS_LIMIT_GB"] = "40"
        env["AURA_MEMWATCH_LETHAL_MB"] = "43008"
        env["AURA_MEMORY_SENTINEL_INTERVAL_S"] = "0.5"
        env["AURA_GOVERNOR_PRUNE_MB"] = "37888"
        env["AURA_GOVERNOR_UNLOAD_MB"] = "39936"
        env["AURA_GOVERNOR_CRITICAL_MB"] = "41984"
        env["AURA_LOCAL_RUNTIME_SINGLETON"] = "1"
        env["AURA_LOCAL_PARALLEL_SLOTS"] = "1"
        env["AURA_ENABLE_LOCAL_DEEP_SOLVER"] = "0"
        env["AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB"] = "24"
        env["AURA_MLX_32B_PROJECTED_FOOTPRINT_GB"] = "auto"
        env["AURA_MLX_32B_PROCESS_RESERVE_GB"] = "3"
        env["AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB"] = "52"
        env["AURA_MLX_72B_PROJECTED_FOOTPRINT_GB"] = "auto"
        env["AURA_MLX_72B_PROCESS_RESERVE_GB"] = "5"
        env["AURA_FOREGROUND_CHAT_MAX_TOKENS"] = "2048"
        env["AURA_EXTERNAL_GUI_OWNER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        env["OBJC_PRINT_LOAD_METHODS"] = "NO"
        for (key, value) in launchProvenanceEnvironment {
            env[key] = value
        }
        return env
    }

    private func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private func normalizedDirectCLIArguments(_ arguments: [String]) -> [String] {
        arguments.map { argument in
            switch argument {
            case "--open-gui-window":
                return "--gui-window"
            default:
                return argument
            }
        }
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
        let provenanceExports = launchProvenanceEnvironment
            .sorted { $0.key < $1.key }
            .map { "export \($0.key)=\(shellQuoted($0.value))" }
            .joined(separator: "\n")
        let helperScript = """
        #!/bin/bash
        cd \(shellQuoted(auraRoot.path))
        \(provenanceExports)
        export AURA_ATTACH_LAUNCHER=0
        export AURA_LOCAL_BACKEND=mlx
        export AURA_LAUNCHED_FROM_APP=1
        export AURA_DESKTOP_RESOURCE_GUARD=1
        export AURA_SAFE_BOOT_DESKTOP=0
        export AURA_ENABLE_BACKGROUND_COGNITION=1
        export AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM=1
        export AURA_BACKGROUND_BOOT_GRACE_S=60
        export AURA_EAGER_LOCAL_SENSORY_BOOT=1
        export AURA_EXTERNAL_GUI_OWNER=1
        export AURA_EAGER_CORTEX_WARMUP=0
        export AURA_DEFERRED_CORTEX_PREWARM=1
        export AURA_DESKTOP_METAL_CACHE_RATIO=0.16
        export AURA_DESKTOP_METAL_CACHE_CAP_GB=10
        export AURA_DESKTOP_MLX_MEMORY_RATIO=0.54
        export AURA_DESKTOP_MLX_MEMORY_CAP_GB=34
        export AURA_DESKTOP_MLX_MEMORY_FLOOR_GB=18
        export AURA_DESKTOP_PROCESS_RSS_RATIO=0.62
        export AURA_DESKTOP_PROCESS_RSS_CAP_GB=40
        export AURA_DESKTOP_PROCESS_RSS_FLOOR_GB=24
        export AURA_PROCESS_RSS_LIMIT_GB=40
        export AURA_MEMWATCH_LETHAL_MB=43008
        export AURA_MEMORY_SENTINEL_INTERVAL_S=0.5
        export AURA_GOVERNOR_PRUNE_MB=37888
        export AURA_GOVERNOR_UNLOAD_MB=39936
        export AURA_GOVERNOR_CRITICAL_MB=41984
        export AURA_LOCAL_RUNTIME_SINGLETON=1
        export AURA_LOCAL_PARALLEL_SLOTS=1
        export AURA_ENABLE_LOCAL_DEEP_SOLVER=0
        export AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB=24
        export AURA_MLX_32B_PROJECTED_FOOTPRINT_GB=auto
        export AURA_MLX_32B_PROCESS_RESERVE_GB=3
        export AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB=52
        export AURA_MLX_72B_PROJECTED_FOOTPRINT_GB=auto
        export AURA_MLX_72B_PROCESS_RESERVE_GB=5
        export AURA_FOREGROUND_CHAT_MAX_TOKENS=2048
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

    /// Turn raw boot blockers into a line a person can read.
    ///
    /// Blockers arrive as internal service ids ("critical:memory_write_gateway").
    /// Joining all of them verbatim made the launcher's most prominent copy a
    /// four-line wall of identifiers. Name the first few in readable form and
    /// count the remainder; the exact list stays in the logs.
    static func blockerSummary(_ blockers: [String], naming limit: Int = 3) -> String {
        // Title-casing every token would render these as "Llm Router", "Api
        // Adapter", "Gwt Winner".
        let acronyms: Set<String> = ["llm", "api", "ui", "gwt", "tts", "stt", "cpu", "ram", "io", "mlx", "vad"]
        let readable = blockers.map { raw -> String in
            let bare = raw.split(separator: ":").last.map(String.init) ?? raw
            return bare
                .replacingOccurrences(of: "_", with: " ")
                .split(separator: " ")
                .map { token -> String in
                    let lower = token.lowercased()
                    if acronyms.contains(lower) { return lower.uppercased() }
                    return token.prefix(1).uppercased() + token.dropFirst()
                }
                .joined(separator: " ")
        }
        guard !readable.isEmpty else { return "" }
        if readable.count <= limit {
            return "waiting on " + readable.joined(separator: ", ")
        }
        let named = readable.prefix(limit).joined(separator: ", ")
        return "waiting on \(named) +\(readable.count - limit) more"
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
                autoOpenDesktopWindowIfNeeded()
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

        if existingRuntimeIsObservable() {
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
            _ = self.forceStopAuraProcess(preserveResidentLauncher: true)
            self.clearBootMarker()
            self.clearTerminalHandoffMarker()
            self.lastSnapshot = nil
            self.forcedRelaunchAttempted = false
            self.autoDesktopOpenTriggered = false
            self.spawnedFreshRuntime = false
            let result = self.launchAuraIfNeeded(forceRelaunch: false)
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
            footer = "Boot phase: \(snapshot.phaseDisplay) • \(Self.blockerSummary(snapshot.blockers))"
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

    /// A start that the runtime positively REFUSED, if one is live.
    ///
    /// The single-instance lock is correct — a second runtime would load a
    /// second copy of the resident model and exhaust the host. What used to be
    /// wrong is that the refusal was invisible: the runtime exited immediately
    /// with EX_TEMPFAIL while this window sat on "waiting for boot health"
    /// forever. acquire_instance_lock now writes the reason here.
    struct BootBlockedNotice {
        let holderPID: Int
        let reason: String
        let remedy: String
        let isBackgroundInstance: Bool
    }

    private func readBootBlockedNotice() -> BootBlockedNotice? {
        guard let file = bootBlockedFile,
              let data = try? Data(contentsOf: file),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pid = json["holder_pid"] as? Int
        else { return nil }

        // A notice naming a process that has since exited is not a live
        // blocker — otherwise stopping the other instance would look like it
        // did nothing.
        guard kill(pid_t(pid), 0) == 0 || errno == EPERM else { return nil }

        return BootBlockedNotice(
            holderPID: pid,
            reason: (json["reason"] as? String) ?? "Another Aura runtime holds the instance lock.",
            remedy: (json["remedy"] as? String) ?? "Stop the other instance, then relaunch.",
            isBackgroundInstance: (json["holder_is_background_instance"] as? Bool) ?? false
        )
    }

    private func renderBootBlocked(_ notice: BootBlockedNotice) {
        renderTitle("Another Aura is already running")
        renderStatus(
            detail: notice.reason,
            footer: notice.remedy,
            progress: 100.0,
            phase: notice.isBackgroundInstance ? "background instance holds the lock" : "instance conflict",
            badgeStyle: .rose,
        )
    }

    private func renderPendingLaunch(waitingOnExisting: Bool) {
        // Never spin on "waking up" when the start was positively refused.
        if let blocked = readBootBlockedNotice() {
            renderBootBlocked(blocked)
            return
        }
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

    private func hideLauncherWindow() {
        window?.orderOut(nil)
    }

    private func desktopWindowIsVisible() -> Bool {
        desktopWindow?.isVisible == true
    }

    private func desktopURL() -> URL {
        let build = bundledSemver.isEmpty ? "live" : bundledSemver
        let ts = Int(Date().timeIntervalSince1970)
        return URL(string: "http://127.0.0.1:8000/?build=\(build)&ts=\(ts)&surface=native-app")!
    }

    // MARK: - Bubble

    private func bubbleURL() -> URL {
        URL(string: "http://127.0.0.1:8000/static/bubble.html?surface=bubble")!
    }

    /// Present the bubble. Idempotent; safe to call on every window close.
    private func showBubble() {
        if bubblePanel == nil {
            // The page reports its measured pill size after load. Starting at
            // the icon footprint prevents an invisible 520pt web view from
            // intercepting clicks over the person's desktop while Aura is idle.
            let size = NSSize(width: 56, height: 56)
            let config = WKWebViewConfiguration()
            // The page talks back through this handler for "open the chat
            // window" — the only intent the bubble has that the web layer
            // cannot serve on its own.
            config.userContentController.add(self, name: "auraBubble")
            let webView = WKWebView(
                frame: NSRect(origin: .zero, size: size), configuration: config
            )
            webView.setValue(false, forKey: "drawsBackground")
            webView.allowsBackForwardNavigationGestures = false
            webView.uiDelegate = self

            let panel = NSPanel(
                contentRect: NSRect(origin: .zero, size: size),
                // .nonactivatingPanel is the whole point: the bubble takes
                // clicks without becoming the active application, so the
                // person's cursor stays where they left it.
                styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
                backing: .buffered,
                defer: false
            )
            panel.isFloatingPanel = true
            panel.level = .floating
            panel.backgroundColor = .clear
            panel.isOpaque = false
            panel.hasShadow = false
            panel.hidesOnDeactivate = false
            panel.isMovableByWindowBackground = true
            panel.collectionBehavior = [
                // Follows the person between Spaces and survives a
                // full-screen app rather than being stranded on one desktop.
                .canJoinAllSpaces,
                .fullScreenAuxiliary,
                .ignoresCycle,
            ]
            panel.contentView = webView
            panel.isReleasedWhenClosed = false
            panel.setFrameOrigin(defaultBubbleOrigin(for: size))
            // No native drag gesture here on purpose. bubble.js drives the
            // bubble's move through {action:"move", relative:true}, and it has
            // to: only the page knows whether the pointer went down on the ×
            // or the reply control, which must stay clickable. Installing a
            // second mechanism would move the panel twice per pointer motion.

            bubblePanel = panel
            bubbleWebView = webView
            // Drag the bubble from ANY point on it, through the same
            // recognizer the companion window uses.
            //
            // LIVE DEFECT, reported three sittings running — "i cant drag the
            // bubble across the screen", then "i still cant drag across the
            // screen", then "the native drag kinda works. it just only works
            // in that upper right quadrant. Not the whole button".
            //
            // The page was doing the dragging: mousedown on the pill, then
            // mousemove deltas posted back as {action:"move"}. That cannot
            // work from a 56x56 web view — WebKit only synthesises mousemove
            // for points inside the view, so the gesture died within about
            // 28px — and whatever part of the glyph the ×/reply controls did
            // not claim was the only surface that responded at all, which is
            // the quadrant.
            //
            // TopStripPanGestureRecognizer already solved this for the
            // companion window, and its own documentation names this window as
            // the reason it takes a strip height of zero: "Zero drags from
            // anywhere, which is what the bubble wants: it is all glyph." It
            // was simply never installed here. One drag mechanism, in global
            // screen coordinates, for both windows.
            //
            // Clicks survive because the recognizer does not delay the primary
            // mouse button and a pan only begins once the pointer actually
            // moves: × and the reply control keep taking plain clicks.
            observeBubbleMoves(panel)
        }

        bubbleWebView?.load(
            URLRequest(url: bubbleURL(), cachePolicy: .reloadIgnoringLocalCacheData)
        )
        // Put her back where she was parked BEFORE showing her. Placing the
        // panel and then moving it would drop her in the default corner and
        // slide her across the screen on every window close.
        //
        // The position was persisted and nothing ever read it back, so
        // "position persists" described a POST with no reader: she reappeared
        // bottom-left after every restart no matter where she had been left.
        restoreBubbleOrigin { [weak self] origin in
            guard let self, let panel = self.bubblePanel else { return }
            if let origin {
                panel.setFrameOrigin(self.clampToScreen(origin, size: panel.frame.size))
            }
            // orderFront, never makeKey: showing her must not take focus.
            panel.orderFront(nil)
            self.matchBackingScale(for: panel)
            // Showing the bubble is also how a person un-hides her, so this
            // has to clear HIDDEN rather than only re-draw the panel.
            self.postAmbientMode("bubble")
        }
    }

    /// Fetch the parked origin, or nil to keep the default corner.
    ///
    /// Always calls back, on the main queue, exactly once — a failed lookup
    /// must still show her. The session's 1s timeout is the bound: a runtime
    /// that is not answering cannot be allowed to mean no bubble at all.
    private func restoreBubbleOrigin(_ done: @escaping (NSPoint?) -> Void) {
        guard let url = URL(string: "http://127.0.0.1:8000/api/ambient/state?surface=restore")
        else {
            DispatchQueue.main.async { done(nil) }
            return
        }
        session.dataTask(with: url) { data, _, _ in
            var origin: NSPoint?
            if let data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let position = json["bubble_position"] as? [Double],
               position.count == 2,
               // (0, 0) is the runtime's "never parked" sentinel, not a
               // corner someone chose.
               position[0] != 0 || position[1] != 0 {
                origin = NSPoint(x: position[0], y: position[1])
            }
            DispatchQueue.main.async { done(origin) }
        }.resume()
    }

    /// Keep her reachable when the display arrangement changed underneath her.
    ///
    /// A position saved on a second monitor that is no longer attached would
    /// otherwise put her somewhere with no pixels, which reads exactly like
    /// the bubble being broken.
    private func clampToScreen(_ origin: NSPoint, size: NSSize) -> NSPoint {
        let visible = (NSScreen.screens.first { $0.visibleFrame.contains(origin) }
            ?? NSScreen.main)?.visibleFrame
        guard let visible else { return origin }
        return NSPoint(
            x: min(max(origin.x, visible.minX), visible.maxX - size.width),
            y: min(max(origin.y, visible.minY), visible.maxY - size.height)
        )
    }

    /// Take the panel off screen WITHOUT changing what she is allowed to do.
    ///
    /// Used when another surface is taking over — opening the full desktop
    /// window retires the bubble, and that is a change of surface, not a
    /// request for her to stop looking. Keeping this separate from hideBubble
    /// is the whole distinction: one is a layout decision, the other is
    /// someone telling her to go away.
    private func orderBubbleOut() {
        bubblePanel?.orderOut(nil)
    }

    /// The person asked her to go away.
    private func hideBubble() {
        orderBubbleOut()
        // Hiding her must stop her LOOKING, not just stop her being drawn.
        // Ordering the panel out on its own left the observation loop reading
        // the screen of someone who had just dismissed her — the one control
        // here that must never be cosmetic.
        postAmbientMode("hidden")
    }

    /// Tell the runtime which surface she is present on.
    ///
    /// The mode is what the observation loop gates on, so it has to follow the
    /// actual window state rather than being assumed. Fire-and-forget on
    /// purpose: this is a notification about a UI transition that has already
    /// happened, and blocking a window close on an HTTP round trip would make
    /// closing a window feel broken. A missed post self-corrects on the next
    /// transition, and the mode it could get stuck in is the SAFE one.
    private func postAmbientMode(_ mode: String) {
        guard let url = URL(string: "http://127.0.0.1:8000/api/ambient/visibility") else {
            return
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{\"mode\":\"\(mode)\"}".utf8)
        session.dataTask(with: request).resume()
    }

    /// Re-rasterise a panel's layers at the scale of the screen it is on.
    ///
    /// Both floating panels are built at origin .zero with `defer: false`,
    /// which means their layers are created before the window has been placed
    /// on a screen. A layer that came up with contentsScale 1.0 keeps it, so
    /// on a Retina display the whole surface — the neuron, the text, the
    /// pill's edge — is drawn at half resolution and then scaled up.
    ///
    /// Reported live 2026-08-10: "the bubble is a little blurry. would like
    /// the image sharpened a bit". The glyph is an SVG and the text is system
    /// text; neither can be blurry on its own. It was the layer underneath
    /// them.
    private func matchBackingScale(for window: NSWindow) {
        let scale = window.screen?.backingScaleFactor
            ?? NSScreen.main?.backingScaleFactor
            ?? 2.0
        guard let root = window.contentView else { return }
        root.wantsLayer = true

        func apply(_ view: NSView) {
            view.layer?.contentsScale = scale
            view.layer?.rasterizationScale = scale
            for child in view.subviews { apply(child) }
        }
        apply(root)
    }

    private func defaultBubbleOrigin(for size: NSSize) -> NSPoint {
        guard let screen = NSScreen.main else { return NSPoint(x: 40, y: 40) }
        let visible = screen.visibleFrame
        // Bottom-left, inset. Out of the way of window controls, menu bar
        // extras, and the notification stack on the right.
        return NSPoint(x: visible.minX + 24, y: visible.minY + 24)
    }

    /// Let a panel whose contentView is a WKWebView be dragged by its content.
    ///
    /// `isMovableByWindowBackground` was set on both panels and neither could
    /// be moved. That flag only fires when the WINDOW receives the background
    /// mouseDown, and a WKWebView consumes every mouse event inside its own
    /// subview tree, so the window never saw one. Reported live 2026-08-10:
    /// "i still cant drag across the screen".
    ///
    /// A pan recognizer sits above the web content and does not delay the
    /// primary mouse button, so a plain click still reaches the page — the
    /// bubble's one job, opening the chat, keeps working — while a drag moves
    /// the host panel.
    /// - Parameter topStrip: points of draggable strip from the top edge, or 0
    ///   for the whole surface. The companion needs a strip so that dragging
    ///   across the transcript still selects text.
    private func installWindowDrag(on view: NSView, topStrip: CGFloat = 0) {
        let pan = TopStripPanGestureRecognizer(
            target: self, action: #selector(handleWindowDrag(_:))
        )
        pan.topStrip = topStrip
        pan.delaysPrimaryMouseButtonEvents = false
        view.addGestureRecognizer(pan)
    }

    @objc private func handleWindowDrag(_ recognizer: NSPanGestureRecognizer) {
        guard let window = recognizer.view?.window else { return }
        switch recognizer.state {
        case .began:
            dragMouseAnchor = NSEvent.mouseLocation
            dragWindowAnchor = window.frame.origin
        case .changed:
            guard let mouseAnchor = dragMouseAnchor,
                  let windowAnchor = dragWindowAnchor else { return }
            let now = NSEvent.mouseLocation
            let moved = NSPoint(
                x: windowAnchor.x + (now.x - mouseAnchor.x),
                y: windowAnchor.y + (now.y - mouseAnchor.y)
            )
            // Clamped so she cannot be dragged off the edge and stranded
            // somewhere with no way to click her back.
            window.setFrameOrigin(clampToScreen(moved, size: window.frame.size))
        case .ended, .cancelled, .failed:
            dragMouseAnchor = nil
            dragWindowAnchor = nil
        default:
            break
        }
    }

    private func observeBubbleMoves(_ panel: NSPanel) {
        bubbleFrameObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didMoveNotification,
            object: panel,
            queue: .main
        ) { [weak self] _ in
            guard let self, let moved = self.bubblePanel else { return }
            let sequence = self.pendingBubbleMoveSequence ?? 0
            self.pendingBubbleMoveSequence = nil
            self.reportBubbleOrigin(moved.frame.origin, sequence: sequence)
        }
    }

    private func reportBubbleOrigin(_ origin: NSPoint, sequence: Int = 0) {
        // The page persists every measured move. A nonzero sequence also
        // closes the cognition-side receipt for an Aura-requested movement.
        let script = """
        window.dispatchEvent(new CustomEvent('aura-bubble-moved', \
        { detail: { x: \(origin.x), y: \(origin.y), sequence: \(sequence) } }));
        """
        bubbleWebView?.evaluateJavaScript(script, completionHandler: nil)
    }

    /// Whether a page asking for the microphone is Aura's own runtime.
    ///
    /// The app holds the microphone TCC grant; this decides which page inside
    /// it may use that grant. Only the local runtime may.
    private func isLocalRuntimeOrigin(_ origin: WKSecurityOrigin) -> Bool {
        let host = origin.host
        let localHost = host == "127.0.0.1" || host == "localhost" || host == "::1"
        return localHost && (origin.protocol == "http" || origin.protocol == "https")
    }

    /// Answer WebKit's microphone/camera prompt for Aura's own pages.
    ///
    /// LIVE DEFECT, 2026-08-10: voice mode showed a red ERROR and never heard
    /// a word, with `[voice] authenticated audio init failed
    /// NotAllowedError: Permission denied` in the page console.
    ///
    /// Nothing was wrong with the microphone. The app has the TCC grant, and
    /// the Python side captures audio through it perfectly — the wake-word
    /// loop was transcribing full sentences at the same time. But voice mode
    /// runs in the WKWebView and asks for `getUserMedia`, and WebKit routes
    /// that request to the app's WKUIDelegate. There was no WKUIDelegate on
    /// any of the three web views, and with no delegate to answer, WebKit
    /// denies. The permission the user had already granted the app could not
    /// reach the page that needed it.
    ///
    /// Granted only for the local runtime: the app's microphone access is not
    /// handed to whatever else a page might load.
    func webView(
        _ webView: WKWebView,
        requestMediaCapturePermissionFor origin: WKSecurityOrigin,
        initiatedByFrame frame: WKFrameInfo,
        type: WKMediaCaptureType,
        decisionHandler: @escaping (WKPermissionDecision) -> Void
    ) {
        guard isLocalRuntimeOrigin(origin) else {
            decisionHandler(.deny)
            return
        }
        decisionHandler(.grant)
    }

    private func openNativeDesktopWindow() {
        if desktopWindow == nil {
            let frame = NSRect(x: 0, y: 0, width: 1280, height: 820)
            let config = WKWebViewConfiguration()
            config.preferences.javaScriptCanOpenWindowsAutomatically = true
            let webView = WKWebView(frame: frame, configuration: config)
            webView.allowsBackForwardNavigationGestures = false
            webView.setValue(false, forKey: "drawsBackground")
            // Without this, voice mode's getUserMedia is denied by default.
            webView.uiDelegate = self

            let desktop = NSWindow(
                contentRect: frame,
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false
            )
            desktop.title = "Aura Zenith"
            desktop.minSize = NSSize(width: 900, height: 640)
            desktop.contentView = webView
            desktop.isReleasedWhenClosed = false
            desktop.delegate = self
            desktopWindow = desktop
            desktopWebView = webView
        }

        desktopWebView?.load(URLRequest(url: desktopURL(), cachePolicy: .reloadIgnoringLocalCacheData))
        // Readiness is a one-window handoff. Retain the monitor for future boot
        // diagnostics, but remove it before presenting the live desktop so a
        // Dock activation cannot leave both windows stacked.
        hideLauncherWindow()
        // The window, the bubble and the restrained chat are the same
        // presence on three surfaces; more than one at once is more than one
        // Aura.
        //
        // orderBubbleOut, NOT hideBubble: opening the full window is a change
        // of surface, and routing it through the dismissal path would tell the
        // runtime she had been sent away every time someone opened her.
        orderBubbleOut()
        hideCompanionChat(restoringBubble: false)
        postAmbientMode("window")
        desktopWindow?.center()
        desktopWindow?.makeKeyAndOrderFront(nil)
        desktopWindow?.orderFrontRegardless()
        NSApp.activate(ignoringOtherApps: true)
        NSRunningApplication.current.activate(options: [])
        NSApp.requestUserAttention(.informationalRequest)
        autoDesktopOpenTriggered = true
        clearGuiWindowLaunchMarker()
    }

    // MARK: - Bubble intents

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard message.name == "auraBubble" || message.name == "auraCompanion" else {
            return
        }
        let body = message.body as? [String: Any] ?? [:]
        switch String(describing: body["action"] ?? "") {
        case "open":
            // Clicking the bubble opens the RESTRAINED window, not the full
            // desktop. Someone who clicked a bubble in the corner wants to
            // say one thing and get back to work; handing them the whole
            // surface makes them close a window they did not ask for.
            showCompanionChat()
        case "expand":
            // They asked for the full desktop explicitly.
            hideCompanionChat(restoringBubble: false)
            openNativeDesktopWindow()
            NSApp.activate(ignoringOtherApps: true)
        case "close":
            hideCompanionChat()
        case "highlight":
            let rect = body["rect"] as? [String: Any] ?? [:]
            showHighlight(
                x: (rect["x"] as? Double) ?? 0,
                y: (rect["y"] as? Double) ?? 0,
                width: (rect["width"] as? Double) ?? 0,
                height: (rect["height"] as? Double) ?? 0,
                seconds: (body["seconds"] as? Double) ?? 3.0
            )
        case "move":
            guard let panel = bubblePanel else { return }
            let sequence = (body["sequence"] as? Int) ?? 0
            // Two callers with different knowledge. Cognition asks her to move
            // to an absolute point it chose. A person dragging her sends a
            // DELTA, because the page cannot read where its own panel sits and
            // guessing an absolute origin from pointer coordinates would fight
            // the clamp on every multi-screen setup.
            let requested: NSPoint
            if (body["relative"] as? Bool) == true {
                requested = NSPoint(
                    x: panel.frame.origin.x + ((body["dx"] as? Double) ?? 0),
                    y: panel.frame.origin.y + ((body["dy"] as? Double) ?? 0)
                )
            } else {
                requested = NSPoint(
                    x: (body["x"] as? Double) ?? panel.frame.origin.x,
                    y: (body["y"] as? Double) ?? panel.frame.origin.y
                )
            }
            pendingBubbleMoveSequence = sequence > 0 ? sequence : nil
            panel.setFrameOrigin(clampToScreen(requested, size: panel.frame.size))
            // AppKit does not emit didMove when the requested origin already
            // equals the clamped origin. Close that no-op command with the
            // measured frame rather than leaving cognition waiting forever.
            DispatchQueue.main.async { [weak self, weak panel] in
                guard let self, let panel,
                      self.pendingBubbleMoveSequence == sequence,
                      sequence > 0 else { return }
                self.pendingBubbleMoveSequence = nil
                self.reportBubbleOrigin(panel.frame.origin, sequence: sequence)
            }
        case "dragStart":
            beginNativeBubbleDrag()
        case "resize":
            resizeBubblePanel(
                width: (body["width"] as? Double) ?? 56,
                height: (body["height"] as? Double) ?? 56
            )
        case "hide":
            hideBubble()
        default:
            break
        }
    }

    /// Make the native hit-test surface exactly as large as visible content.
    ///
    /// CSS pointer-events do not make transparent AppKit window area disappear;
    /// the NSPanel itself can still sit over another app. Resizing the host is
    /// therefore a functional desktop contract, not a visual optimization.
    /// Drag the bubble with AppKit's own tracking loop.
    ///
    /// The companion window is dragged by TopStripPanGestureRecognizer, and
    /// this window was moved onto it for the sake of having ONE mechanism.
    /// That was wrong, and the report was "pretty sure this icon in companion
    /// mode stopped being draggable": the two windows are not the same kind of
    /// window. The bubble is a .nonactivatingPanel — the whole point of it is
    /// that clicking never steals focus — and AppKit does not deliver the drag
    /// to a gesture recognizer on one the way it does on the companion's
    /// keyable panel. Unifying on the recognizer removed the JS drag, the
    /// recognizer never fired here, and nothing was left.
    ///
    /// trackEvents does not depend on gesture recognition at all. It runs a
    /// modal loop that receives every drag event wherever the pointer goes,
    /// and NSEvent.mouseLocation is in global screen coordinates, so the 56x56
    /// bounds of this web view stop mattering. That is what makes the WHOLE
    /// button draggable rather than whichever corner the controls do not
    /// claim.
    ///
    /// Click and drag are separated HERE rather than in the page, because once
    /// this loop starts the page never sees the mouseUp — the loop dequeues
    /// it. A gesture under the threshold is reported back as a click so
    /// tapping the bubble still opens the chat.
    private func beginNativeBubbleDrag() {
        guard let panel = bubblePanel else { return }
        let startMouse = NSEvent.mouseLocation
        let startOrigin = panel.frame.origin
        var moved = false

        panel.trackEvents(
            matching: [.leftMouseDragged, .leftMouseUp],
            timeout: NSEvent.foreverDuration,
            mode: .eventTracking
        ) { event, stop in
            guard let event else {
                stop.pointee = true
                return
            }
            if event.type == .leftMouseUp {
                stop.pointee = true
                return
            }
            let now = NSEvent.mouseLocation
            let dx = now.x - startMouse.x
            let dy = now.y - startMouse.y
            // A few pixels of slop so a click with a shaky hand stays a click.
            if !moved && abs(dx) < 3 && abs(dy) < 3 { return }
            moved = true
            let target = NSPoint(x: startOrigin.x + dx, y: startOrigin.y + dy)
            panel.setFrameOrigin(
                self.clampToScreen(target, size: panel.frame.size)
            )
        }

        if moved {
            reportBubbleOrigin(panel.frame.origin)
        } else {
            bubbleWebView?.evaluateJavaScript(
                "window.dispatchEvent(new CustomEvent('aura-bubble-click'));",
                completionHandler: nil
            )
        }
    }

    private func resizeBubblePanel(width: Double, height: Double) {
        guard let panel = bubblePanel else { return }
        let target = NSSize(
            width: max(48, min(520, width)),
            height: max(48, min(190, height))
        )
        guard abs(panel.frame.width - target.width) > 0.5
                || abs(panel.frame.height - target.height) > 0.5 else { return }
        let origin = clampToScreen(panel.frame.origin, size: target)
        panel.setFrame(NSRect(origin: origin, size: target), display: true, animate: false)
    }

    // MARK: - Highlight overlay

    /// Draw a transient, CLICK-THROUGH rectangle over the screen.
    ///
    /// ignoresMouseEvents is the safety property, not a nicety: an overlay
    /// that can swallow a click can cost someone work, and this window sits
    /// over whatever they were doing. It also auto-dismisses — a highlight
    /// that outlives its answer becomes furniture on someone's desktop.
    func showHighlight(x: Double, y: Double, width: Double, height: Double, seconds: Double) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.overlayDismissWork?.cancel()

            // Accessibility reports top-left origin; AppKit windows are
            // bottom-left. Converting wrong puts the rectangle around
            // something else entirely, which is worse than not drawing it.
            let screenHeight = NSScreen.main?.frame.height ?? 0
            let frame = NSRect(
                x: x,
                y: screenHeight - y - height,
                width: max(4, width),
                height: max(4, height)
            )

            let window = self.overlayWindow ?? {
                let created = NSWindow(
                    contentRect: frame,
                    styleMask: [.borderless],
                    backing: .buffered,
                    defer: false
                )
                created.isOpaque = false
                created.backgroundColor = .clear
                created.hasShadow = false
                created.ignoresMouseEvents = true
                created.level = .screenSaver
                created.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
                created.isReleasedWhenClosed = false
                let view = NSView(frame: NSRect(origin: .zero, size: frame.size))
                view.wantsLayer = true
                created.contentView = view
                self.overlayWindow = created
                return created
            }()

            window.setFrame(frame, display: false)
            window.contentView?.frame = NSRect(origin: .zero, size: frame.size)
            if let layer = window.contentView?.layer {
                layer.borderWidth = 2.5
                layer.borderColor = NSColor(
                    calibratedRed: 0.62, green: 0.45, blue: 1.0, alpha: 0.95
                ).cgColor
                layer.cornerRadius = 6
                layer.backgroundColor = NSColor(
                    calibratedRed: 0.62, green: 0.45, blue: 1.0, alpha: 0.14
                ).cgColor
            }
            // orderFront, never makeKey: pointing at something must not take
            // focus away from it.
            window.orderFront(nil)

            let dismiss = DispatchWorkItem { [weak self] in
                self?.overlayWindow?.orderOut(nil)
            }
            self.overlayDismissWork = dismiss
            DispatchQueue.main.asyncAfter(
                deadline: .now() + max(0.5, min(10.0, seconds)), execute: dismiss
            )
        }
    }

    private func companionURL() -> URL {
        URL(string: "http://127.0.0.1:8000/static/companion_chat.html?surface=companion")!
    }

    private func showCompanionChat() {
        if companionPanel == nil {
            let size = NSSize(width: 420, height: 380)
            let config = WKWebViewConfiguration()
            config.userContentController.add(self, name: "auraCompanion")
            let webView = WKWebView(
                frame: NSRect(origin: .zero, size: size), configuration: config
            )
            webView.setValue(false, forKey: "drawsBackground")
            webView.uiDelegate = self

            let panel = KeyablePanel(
                contentRect: NSRect(origin: .zero, size: size),
                // Titled would give it a chrome bar this window does not
                // want, and .nonactivatingPanel is wrong HERE: unlike the
                // bubble, this window exists to be typed into, and a text
                // field you cannot focus is not a chat window.
                //
                // Omitting .nonactivatingPanel was necessary and not
                // sufficient. AppKit returns canBecomeKey == false for a
                // .borderless window regardless of style mask, so
                // makeKeyAndOrderFront below never actually made it key and
                // the composer could not take a keystroke — reported live,
                // 2026-08-10: "i cant type in the mini chat". KeyablePanel
                // overrides exactly that one property.
                styleMask: [.borderless, .fullSizeContentView, .resizable],
                backing: .buffered,
                defer: false
            )
            panel.isFloatingPanel = true
            panel.level = .floating
            panel.backgroundColor = .clear
            panel.isOpaque = false
            panel.hasShadow = true
            panel.isMovableByWindowBackground = true
            // NSPanel defaults hidesOnDeactivate to TRUE — unlike NSWindow.
            // Unset, every click into another app ordered the companion out,
            // which reads as the window closing itself. Reported live
            // 2026-08-10: "when i click on another window, the companion goes
            // away". The bubble sets this explicitly; this window never did.
            panel.hidesOnDeactivate = false
            panel.minSize = NSSize(width: 340, height: 260)
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            panel.contentView = webView
            panel.isReleasedWhenClosed = false
            panel.setFrameOrigin(companionOrigin(for: size))
            // 37pt: the title bar's 8pt padding, 20pt glyph, 8pt padding and
            // its 1pt rule — the strip the page styles as its drag handle.
            installWindowDrag(on: webView, topStrip: 37)

            companionPanel = panel
            companionWebView = webView
        }

        // Keep one loaded page for the life of the native host. Reloading on
        // every open erased the restrained transcript and created a fresh chat
        // session, so a follow-up could lose the thing it referred to.
        if companionWebView?.url == nil {
            companionWebView?.load(
                URLRequest(url: companionURL(), cachePolicy: .reloadIgnoringLocalCacheData)
            )
        }
        orderBubbleOut()
        postAmbientMode("window")
        // makeKey, because this one is for typing into. KeyablePanel is what
        // lets that actually take effect on a borderless window.
        companionPanel?.makeKeyAndOrderFront(nil)
        if let panel = companionPanel { matchBackingScale(for: panel) }
        NSApp.activate(ignoringOtherApps: true)
        postCompanionVisibility(true)
    }

    private func hideCompanionChat(restoringBubble: Bool = true) {
        companionPanel?.orderOut(nil)
        // Ordering a WKWebView out does not make the page "hidden" by any
        // measure the page can take for itself, and it keeps running: a turn
        // started here and then collapsed is answered into a window nobody is
        // looking at. Only the host knows, so only the host can say.
        postCompanionVisibility(false)
        if restoringBubble && !desktopWindowIsVisible() {
            showBubble()
        }
    }

    private func postCompanionVisibility(_ visible: Bool) {
        let script = """
        window.dispatchEvent(new CustomEvent('aura-companion-visibility', \
        { detail: { visible: \(visible ? "true" : "false") } }));
        """
        companionWebView?.evaluateJavaScript(script, completionHandler: nil)
    }

    private func companionOrigin(for size: NSSize) -> NSPoint {
        // Directly above the bubble when there is one, so the window opens
        // where the person's attention already is rather than centring on a
        // screen they were not looking at.
        if let bubble = bubblePanel {
            let frame = bubble.frame
            return NSPoint(x: frame.minX, y: frame.maxY + 10)
        }
        guard let screen = NSScreen.main else { return NSPoint(x: 60, y: 60) }
        let visible = screen.visibleFrame
        return NSPoint(x: visible.minX + 24, y: visible.minY + 100)
    }

    func windowWillClose(_ notification: Notification) {
        guard (notification.object as? NSWindow) === desktopWindow else { return }
        // She does not leave when the window does. Closing the window is a
        // request for less surface area, not for her to stop; the bubble is
        // what "still here, out of the way" looks like.
        showBubble()
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

    private func guiWindowLaunchAge() -> TimeInterval? {
        guard let text = try? String(contentsOf: guiWindowMarkerFile, encoding: .utf8) else {
            return nil
        }
        guard let epoch = Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return nil
        }
        return Date().timeIntervalSince1970 - epoch
    }

    private func guiWindowLaunchIsFresh() -> Bool {
        guard let age = guiWindowLaunchAge() else {
            return false
        }
        return age >= 0 && age < guiWindowLaunchWindow
    }

    private func markGuiWindowLaunch() {
        let payload = "\(Date().timeIntervalSince1970)\n"
        try? payload.write(to: guiWindowMarkerFile, atomically: true, encoding: .utf8)
    }

    private func clearGuiWindowLaunchMarker() {
        try? fileManager.removeItem(at: guiWindowMarkerFile)
    }

    private func guiWindowHelperIsRunning() -> Bool {
        spawnedProcessesLock.lock()
        defer { spawnedProcessesLock.unlock() }
        return spawnedProcesses.contains { proc in
            proc.isRunning && (proc.arguments ?? []).contains("--gui-window")
        }
    }

    private func desktopWindowLaunchInProgress() -> Bool {
        desktopWindowIsVisible() || guiWindowHelperIsRunning() || guiWindowLaunchIsFresh()
    }

    private func runtimeLockFileURL() -> URL {
        lockDirectory.appendingPathComponent("orchestrator.lock")
    }

    private func parseRuntimeLockPID(_ text: String) -> Int32? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }
        if let firstLine = trimmed.split(whereSeparator: \.isNewline).first,
           let pid = Int32(String(firstLine).trimmingCharacters(in: .whitespacesAndNewlines)),
           pid > 0 {
            return pid
        }
        guard let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let payload = object as? [String: Any] else {
            return nil
        }
        if let pid = payload["pid"] as? Int, pid > 0 {
            return Int32(pid)
        }
        if let pid = payload["pid"] as? String,
           let parsed = Int32(pid),
           parsed > 0 {
            return parsed
        }
        return nil
    }

    private func runtimeLockIndicatesLiveProcess() -> Bool {
        let lockFile = runtimeLockFileURL()
        guard let text = try? String(contentsOf: lockFile, encoding: .utf8) else {
            return false
        }
        guard let pid = parseRuntimeLockPID(text) else {
            return false
        }
        return kill(pid, 0) == 0 || errno == EPERM
    }

    private func fetchBootSnapshotSynchronously(timeout: TimeInterval = 1.2) -> BootSnapshot? {
        let request = URLRequest(
            url: URL(string: "http://127.0.0.1:8000/api/health/boot")!,
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: timeout
        )
        let semaphore = DispatchSemaphore(value: 0)
        var snapshot: BootSnapshot?
        let task = session.dataTask(with: request) { data, response, _ in
            snapshot = Self.parseSnapshot(data: data, response: response)
            semaphore.signal()
        }
        task.resume()
        if semaphore.wait(timeout: .now() + timeout + 0.2) == .timedOut {
            task.cancel()
            return nil
        }
        return snapshot
    }

    private func existingRuntimeIsObservable() -> Bool {
        guard runtimeLockIndicatesLiveProcess() else {
            return false
        }
        guard let snapshot = fetchBootSnapshotSynchronously() else {
            // A long foreground generation can miss the short launcher poll.
            // Preserve the lock when health is merely unavailable; only an
            // explicit boot contract failure is allowed to trigger cleanup.
            return true
        }
        if let reason = snapshot.staleRuntimeFailureReason
            ?? snapshot.replacementReason(expectedSemver: bundledSemver) {
            _ = forceStopAuraProcess(preserveResidentLauncher: true)
            clearBootMarker()
            clearTerminalHandoffMarker()
            lastSnapshot = nil
            forcedRelaunchAttempted = false
            autoDesktopOpenTriggered = false
            spawnedFreshRuntime = false
            DispatchQueue.main.async { [weak self] in
                self?.footerLabel.stringValue = "Replaced stale Aura runtime before launch: \(reason)"
            }
            return false
        }
        return true
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
            // Health polling can time out while the Python event loop is busy
            // generating a long reply. The orchestrator PID lock is the
            // authoritative process-liveness signal; never spawn a second
            // kernel merely because one HTTP poll missed its one-second SLA.
            if !forceRelaunch && self.existingRuntimeIsObservable() {
                return .observingExistingBoot
            }
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
        hideLauncherWindow()
    }


    @objc private func forceStopAura() {
        explicitStopInProgress = true
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

    private func forceStopAuraProcess(preserveResidentLauncher: Bool = false) -> String {
        explicitStopInProgress = true
        let cleanupScript = auraRoot.appendingPathComponent("aura_cleanup.py")
        let logHandle: FileHandle?
        do {
            logHandle = try openLogHandle()
        } catch {
            return "Aura stop failed before cleanup logging could start: \(error.localizedDescription)"
        }

        func runTool(arguments: [String], timeout: TimeInterval = 45.0) -> Bool {
            let proc = Process()
            proc.executableURL = pythonExecutable
            proc.arguments = arguments
            proc.currentDirectoryURL = auraRoot
            var env = baseAuraEnvironment()
            if preserveResidentLauncher {
                env["AURA_STOP_PRESERVE_RESIDENT_LAUNCHER"] = "1"
                env["AURA_STOP_GRACE_SECONDS"] = env["AURA_STOP_GRACE_SECONDS"] ?? "18"
            }
            proc.environment = env
            proc.standardOutput = logHandle
            proc.standardError = logHandle
            do {
                try proc.run()
                let deadline = Date().addingTimeInterval(timeout)
                while proc.isRunning && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.05)
                }
                if proc.isRunning {
                    if let data = "Aura stop helper timed out after \(timeout)s: \(arguments.joined(separator: " "))\n"
                        .data(using: .utf8) {
                        logHandle?.write(data)
                    }
                    proc.terminate()
                    let terminateDeadline = Date().addingTimeInterval(2.0)
                    while proc.isRunning && Date() < terminateDeadline {
                        Thread.sleep(forTimeInterval: 0.05)
                    }
                    if proc.isRunning {
                        kill(proc.processIdentifier, SIGKILL)
                    }
                    return false
                }
                return proc.terminationStatus == 0
            } catch {
                return false
            }
        }

        let stopOK = runTool(arguments: ["-u", auraMainScript.path, "--stop"])
        let cleanupOK = fileManager.fileExists(atPath: cleanupScript.path)
            ? runTool(arguments: [cleanupScript.path], timeout: 20.0)
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
        if desktopWindowLaunchInProgress() {
            footerLabel.stringValue = "Aura’s desktop window is already open or opening. I’m keeping this launch single-flight."
            if desktopWindowIsVisible() {
                openNativeDesktopWindow()
            }
            return
        }
        if existingRuntimeIsObservable() {
            openNativeDesktopWindow()
            return
        }

        footerLabel.stringValue = "Aura is still booting. The desktop window will open when the live runtime is observable."
    }

    @objc private func openBrowser() {
        let build = bundledSemver.isEmpty ? "live" : bundledSemver
        let ts = Int(Date().timeIntervalSince1970)
        if let url = URL(string: "http://127.0.0.1:8000/?build=\(build)&ts=\(ts)") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func closeLauncher() {
        hideLauncherWindow()
    }

    private func spawnAuxiliaryAura(arguments: [String]) throws {
        let directArguments = ["-u", auraMainScript.path] + normalizedDirectCLIArguments(arguments)
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
        trackSpawnedProcess(proc)
    }

    private func trackSpawnedProcess(_ proc: Process) {
        let pid = proc.processIdentifier
        spawnedProcessesLock.lock()
        spawnedProcesses.removeAll { $0.processIdentifier == pid }
        spawnedProcesses.append(proc)
        spawnedProcessesLock.unlock()
        proc.terminationHandler = { [weak self] finished in
            self?.untrackSpawnedProcess(pid: finished.processIdentifier)
        }
    }

    private func untrackSpawnedProcess(pid: Int32) {
        spawnedProcessesLock.lock()
        spawnedProcesses.removeAll { $0.processIdentifier == pid }
        spawnedProcessesLock.unlock()
    }

    private func terminateSpawnedProcesses() {
        spawnedProcessesLock.lock()
        let processes = spawnedProcesses
        spawnedProcesses.removeAll()
        spawnedProcessesLock.unlock()

        for proc in processes where proc.isRunning {
            proc.terminate()
        }

        let configuredGrace = ProcessInfo.processInfo.environment["AURA_LAUNCHER_CHILD_TERMINATION_GRACE_S"]
            .flatMap(TimeInterval.init) ?? 18.0
        let deadline = Date().addingTimeInterval(min(30.0, max(5.0, configuredGrace)))
        for proc in processes where proc.isRunning {
            while proc.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
                let reapDeadline = Date().addingTimeInterval(2.0)
                while proc.isRunning && Date() < reapDeadline {
                    Thread.sleep(forTimeInterval: 0.05)
                }
            }
        }
    }

    @discardableResult
    private func autoOpenDesktopWindowIfNeeded() -> Bool {
        if autoDesktopOpenTriggered {
            return true
        }
        if terminalHandoffIsFresh() {
            return true
        }
        if desktopWindowLaunchInProgress() {
            autoDesktopOpenTriggered = true
            if desktopWindowIsVisible() {
                openNativeDesktopWindow()
            }
            return true
        }
        openNativeDesktopWindow()
        return true
    }
}

if let bridgeIndex = CommandLine.arguments.firstIndex(of: nativeBridgeFlag) {
    let payloadIndex = CommandLine.arguments.index(after: bridgeIndex)
    guard payloadIndex < CommandLine.arguments.endIndex,
          let data = CommandLine.arguments[payloadIndex].data(using: .utf8),
          let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        bridgeJSON(["ok": false, "error": "invalid_bridge_payload"], status: 2)
    }
    runNativeDesktopBridge(payload: payload)
} else {
    let app = NSApplication.shared
    let delegate = AuraLauncherDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.regular)
    app.run()
}
