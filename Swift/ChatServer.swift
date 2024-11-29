import Foundation

// Actor to manage shared chat state
actor ChatRoom {
    private var messages: [String] = []
    
    func broadcast(message: String) {
        messages.append(message)
        print(message)
    }
    
    func getAllMessages() -> [String] {
        return messages
    }
}

// Represents a chat user
class ChatUser {
    let id: Int
    private let chatRoom: ChatRoom
    private let messageOptions = [
        "Hello everyone!",
        "How's it going?",
        "Nice weather today!",
        "Anyone want to chat?",
        "I'm having a great day!",
        "What's new?",
        "Just checking in..."
    ]
    
    init(id: Int, chatRoom: ChatRoom) {
        self.id = id
        self.chatRoom = chatRoom
    }
    
    func generateMessage() -> String {
        let randomMessage = messageOptions.randomElement() ?? "Hello!"
        return "User \(id): \(randomMessage)"
    }
    
    func startChatting() async {
        while true {
            let message = generateMessage()
            await chatRoom.broadcast(message: message)
            
            // Random delay between messages (1-5 seconds)
            try? await Task.sleep(nanoseconds: UInt64.random(in: 1_000_000_000...5_000_000_000))
        }
    }
}

// Main chat server simulation
func runChatServer() async {
    let chatRoom = ChatRoom()
    let numberOfUsers = 5
    
    // Create multiple concurrent user tasks
    let users = (1...numberOfUsers).map { ChatUser(id: $0, chatRoom: chatRoom) }
    
    // Start all users chatting concurrently
    await withTaskGroup(of: Void.self) { group in
        for user in users {
            group.addTask {
                await user.startChatting()
            }
        }
        
        // This will run indefinitely
        await group.waitForAll()
    }
}

// Entry point
Task {
    await runChatServer()
}

// Keep the program running
RunLoop.main.run()