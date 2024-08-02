import Foundation

protocol Therapist {
    var name: String { get }
    var specialization: String { get }
    var clients: [Client] { get set }
    
    func addClient(_ client: Client)
    func conductSession(with client: Client) throws
    func createTreatmentPlan(for client: Client) -> TreatmentPlan
    func generateSessionReport(for client: Client, session: TherapySession) -> String
}

protocol TherapeuticTechnique {
    func apply(to client: Client)
}

struct Client {
    let id: UUID
    var name: String
    var age: Int
    var issues: [String]
    var sessionHistory: [SessionRecord]
    var treatmentPlan: TreatmentPlan?
    
    init(name: String, age: Int, issues: [String]) {
        self.id = UUID()
        self.name = name
        self.age = age
        self.issues = issues
        self.sessionHistory = []
    }
}

struct SessionRecord {
    let date: Date
    let notes: String
    let rating: Int
    let feedback: String
}

struct TreatmentPlan {
    var goals: [String]
    var techniques: [TherapeuticTechnique]
}

struct BreathingExercise: TherapeuticTechnique {
    func apply(to client: Client) {
        print("Guiding \(client.name) through breathing exercises.")
    }
}

struct GratitudeJournaling: TherapeuticTechnique {
    func apply(to client: Client) {
        print("Encouraging \(client.name) to maintain a gratitude journal.")
    }
}

struct StressManagement: TherapeuticTechnique {
    func apply(to client: Client) {
        print("Helping \(client.name) identify stressors and develop coping strategies.")
    }
}

class Psychologist: Therapist {
    let name: String
    let specialization: String
    var clients: [Client]
    
    init(name: String, specialization: String) {
        self.name = name
        self.specialization = specialization
        self.clients = []
    }
    
    func addClient(_ client: Client) {
        clients.append(client)
    }
    
    func conductSession(with client: Client) throws {
        guard let clientIndex = clients.firstIndex(where: { $0.id == client.id }) else {
            throw TherapyError.clientNotFound
        }
        
        print("Starting session with \(client.name)")
        
        if clients[clientIndex].treatmentPlan == nil {
            clients[clientIndex].treatmentPlan = createTreatmentPlan(for: client)
        }
        
        guard let treatmentPlan = clients[clientIndex].treatmentPlan else {
            throw TherapyError.treatmentPlanNotFound
        }
        
        for technique in treatmentPlan.techniques {
            technique.apply(to: client)
        }
        
        print("Session with \(client.name) completed")
    }
    
    func createTreatmentPlan(for client: Client) -> TreatmentPlan {
        var techniques: [TherapeuticTechnique] = []
        
        for issue in client.issues {
            switch issue.lowercased() {
            case "anxiety":
                techniques.append(BreathingExercise())
            case "depression":
                techniques.append(GratitudeJournaling())
            case "stress":
                techniques.append(StressManagement())
            default:
                break
            }
        }
        
        return TreatmentPlan(goals: ["Reduce symptoms", "Improve coping skills"], techniques: techniques)
    }
    
    func generateSessionReport(for client: Client, session: TherapySession) -> String {
        return """
        Session Report:
        Client: \(client.name)
        Date: \(Date())
        Duration: \(session.duration) minutes
        Techniques applied: \(session.techniques.map { String(describing: type(of: $0)) }.joined(separator: ", "))
        Notes: Session conducted successfully.
        """
    }
}

class TherapySession {
    let id: UUID
    var psychologist: Therapist
    var client: Client
    let duration: Int
    var techniques: [TherapeuticTechnique]
    var rating: Int?
    var feedback: String?
    
    init(psychologist: Therapist, client: Client, duration: Int) {
        self.id = UUID()
        self.psychologist = psychologist
        self.client = client
        self.duration = duration
        self.techniques = []
    }
    
    func start() throws {
        print("Starting therapy session with \(client.name) and \(psychologist.name)")
        try psychologist.conductSession(with: client)
        applyTechniques()
        endSession()
    }
    
    private func applyTechniques() {
        for technique in techniques {
            technique.apply(to: client)
        }
    }
    
    private func endSession() {
        print("Therapy session with \(client.name) has ended after \(duration) minutes")
        requestFeedback()
    }
    
    private func requestFeedback() {
        print("Please rate the session from 1 to 5:")
        rating = Int.random(in: 1...5)  // Simulating user input
        print("Please provide any feedback:")
        feedback = "Great session!"  // Simulating user input
        
        if let clientIndex = psychologist.clients.firstIndex(where: { $0.id == client.id }) {
            let sessionRecord = SessionRecord(date: Date(), notes: psychologist.generateSessionReport(for: client, session: self), rating: rating ?? 0, feedback: feedback ?? "")
            psychologist.clients[clientIndex].sessionHistory.append(sessionRecord)
        }
    }
}

enum TherapyError: Error {
    case clientNotFound
    case treatmentPlanNotFound
}

let drSmith = Psychologist(name: "Dr. Smith", specialization: "Cognitive Behavioral Therapy")
let johnDoe = Client(name: "John Doe", age: 35, issues: ["anxiety", "stress"])
drSmith.addClient(johnDoe)

let session = TherapySession(psychologist: drSmith, client: johnDoe, duration: 60)
session.techniques = [BreathingExercise(), StressManagement()]

do {
    try session.start()
} catch {
    print("An error occurred: \(error)")
}