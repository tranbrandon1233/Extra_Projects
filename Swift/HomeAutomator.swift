import Foundation

protocol SmartDevice {
    var id: String { get }
    var name: String { get }
    var isOnline: Bool { get }
    var lastUpdated: Date { get }
}

protocol Switchable {
    var isOn: Bool { get set }
    func toggle()
}

protocol TemperatureControl {
    var currentTemperature: Double { get }
    var targetTemperature: Double { get set }
}

protocol VideoCapture {
    var isRecording: Bool { get set }
    func startRecording()
    func stopRecording()
    func takeSnapshot() -> Data?
}

class SmartLight: SmartDevice, Switchable {
    let id: String
    var name: String
    var isOnline: Bool
    var lastUpdated: Date
    var isOn: Bool
    var brightness: Int
    
    init(withId id: String, withName name: String) {
        self.id = id
        self.name = name
        self.isOnline = true
        self.lastUpdated = Date()
        self.isOn = false
        self.brightness = 0
        print("SmartLight initialized: \(name)")
    }
    
    func toggle() {
        isOn.toggle()
        lastUpdated = Date()
        print("[\(name)] Power state changed to: \(isOn ? "ON" : "OFF")")
    }
    
    func setBrightness(to level: Int) {
        brightness = min(max(level, 0), 100)
        lastUpdated = Date()
        print("[\(name)] Brightness set to: \(brightness)%")
    }
}

class SmartThermostat: SmartDevice, TemperatureControl {
    let id: String
    var name: String
    var isOnline: Bool
    var lastUpdated: Date
    var currentTemperature: Double
    var targetTemperature: Double {
        didSet {
            print("[\(name)] Target temperature changed to: \(targetTemperature)°C")
        }
    }
    var mode: ThermostatMode
    
    enum ThermostatMode {
        case off, heat, cool, auto
    }
    
    init(withId id: String, withName name: String) {
        self.id = id
        self.name = name
        self.isOnline = true
        self.lastUpdated = Date()
        self.currentTemperature = 20.0
        self.targetTemperature = 20.0
        self.mode = .off
        print("SmartThermostat initialized: \(name)")
    }
    
    func setMode(to mode: ThermostatMode) {
        self.mode = mode
        lastUpdated = Date()
        print("[\(name)] Mode set to: \(mode)")
    }
}

class SecurityCamera: SmartDevice, VideoCapture {
    let id: String
    var name: String
    var isOnline: Bool
    var lastUpdated: Date
    var isRecording: Bool
    
    init(withId id: String, withName name: String) {
        self.id = id
        self.name = name
        self.isOnline = true
        self.lastUpdated = Date()
        self.isRecording = false
        print("SecurityCamera initialized: \(name)")
    }
    
    func startRecording() {
        isRecording = true
        lastUpdated = Date()
        print("[\(name)] Started recording")
    }
    
    func stopRecording() {
        isRecording = false
        lastUpdated = Date()
        print("[\(name)] Stopped recording")
    }
    
    func takeSnapshot() -> Data? {
        lastUpdated = Date()
        print("[\(name)] Snapshot taken")
        return "Snapshot data".data(using: .utf8)
    }
}

class SmartHomeManager {
    static let shared = SmartHomeManager()
    
    private var devices: [String: SmartDevice] = [:]
    private var automationRules: [AutomationRule] = []
    
    private init() {
        print("SmartHomeManager initialized")
    }
    
    func addDevice(device: SmartDevice) {
        devices[device.id] = device
        let deviceType = type(of: device)
        print("[\(deviceType)] Added to manager: \(device.name) [\(device.id)]")
    }
    
    func removeDevice(withId id: String) {
        if let device = devices.removeValue(forKey: id) {
            let deviceType = type(of: device)
            print("[\(deviceType)] Removed from manager: \(device.name) [\(device.id)]")
        }
    }
    
    func getDevice(withId id: String) -> SmartDevice? {
        return devices[id]
    }
    
    func getAllDevices() -> [SmartDevice] {
        return Array(devices.values)
    }
    
    func addAutomationRule(rule: AutomationRule) {
        automationRules.append(rule)
        print("New automation rule added: '\(rule.name)'")
    }
    
    func executeAutomationRules() {
        print("Starting automation rules execution...")
        for rule in automationRules {
            rule.evaluate()
        }
        print("Completed automation rules execution")
    }
}

struct AutomationRule {
    let name: String
    let condition: () -> Bool
    let action: () -> Void
    
    func evaluate() {
        print("Evaluating rule: '\(name)'")
        if condition() {
            print("Condition met for rule: '\(name)'. Executing action...")
            action()
            print("Completed execution of rule: '\(name)'")
        } else {
            print("Condition not met for rule: '\(name)'")
        }
    }
}

class RemoteAccess {
    static let shared = RemoteAccess()
    
    private var isConnected: Bool = false
    private var lastConnectionAttempt: Date?
    
    private init() {
        print("RemoteAccess system initialized")
    }
    
    func connect() -> Bool {
        isConnected = true
        lastConnectionAttempt = Date()
        print("Remote access connection established")
        return isConnected
    }
    
    func disconnect() {
        isConnected = false
        print("Remote access disconnected")
    }
    
    func executeCommand(command: RemoteCommand) -> Bool {
        guard isConnected else {
            print("Command failed: Remote access not connected")
            return false
        }
        
        print("Executing remote command: \(command)")
        
        switch command {
        case .toggle(let deviceId):
            if let device = SmartHomeManager.shared.getDevice(withId: deviceId) as? Switchable {
                device.toggle()
                print("Successfully executed toggle command for device: \(deviceId)")
                return true
            }
        case .setTemperature(let deviceId, let temperature):
            if let device = SmartHomeManager.shared.getDevice(withId: deviceId) as? SmartThermostat {
                device.targetTemperature = temperature
                print("Successfully executed temperature change command for device: \(deviceId)")
                return true
            }
        case .startRecording(let deviceId):
            if let device = SmartHomeManager.shared.getDevice(withId: deviceId) as? SecurityCamera {
                device.startRecording()
                print("Successfully executed recording command for device: \(deviceId)")
                return true
            }
        }
        
        print("Command execution failed: Device not found or incompatible command")
        return false
    }
}

enum RemoteCommand: CustomStringConvertible {
    case toggle(deviceId: String)
    case setTemperature(deviceId: String, temperature: Double)
    case startRecording(deviceId: String)
    
    var description: String {
        switch self {
        case .toggle(let deviceId):
            return "Toggle device [\(deviceId)]"
        case .setTemperature(let deviceId, let temperature):
            return "Set temperature for device [\(deviceId)] to \(temperature)°C"
        case .startRecording(let deviceId):
            return "Start recording for device [\(deviceId)]"
        }
    }
}

func setupExampleSmartHome() {
    print("\n=== Setting up Smart Home System ===\n")
    
    let manager = SmartHomeManager.shared
    
    let livingRoomLight = SmartLight(withId: "light1", withName: "Living Room Light")
    let thermostat = SmartThermostat(withId: "therm1", withName: "Main Thermostat")
    let frontCamera = SecurityCamera(withId: "cam1", withName: "Front Door Camera")
    
    manager.addDevice(device: livingRoomLight)
    manager.addDevice(device: thermostat)
    manager.addDevice(device: frontCamera)
    
    let eveningRule = AutomationRule(
        name: "Evening Lights",
        condition: {
            let hour = Calendar.current.component(.hour, from: Date())
            return hour >= 18
        },
        action: {
            if let light = manager.getDevice(withId: "light1") as? SmartLight {
                light.isOn = true
                light.setBrightness(to: 70)
            }
        }
    )
    
    manager.addAutomationRule(rule: eveningRule)
    
    print("\n=== Testing Remote Access ===\n")
    
    let remote = RemoteAccess.shared
    if remote.connect() {
        let _ = remote.executeCommand(command: .toggle(deviceId: "light1"))
        let _ = remote.executeCommand(command: .setTemperature(deviceId: "therm1", temperature: 22.5))
        
    }
    
    print("\n=== Setup Complete ===\n")
}

setupExampleSmartHome()