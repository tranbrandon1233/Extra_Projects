import Foundation

/// Represents a user in the chatroom.
class User {
    let id: UUID
    var username: String
    var flair: String?
    var isModerator: Bool
    var isBanned: Bool
    var timeoutEndDate: Date?
    
    /// Initializes a new user with the given username.
    /// - Parameter username: The username of the user.
    init(username: String) {
        self.id = UUID()
        self.username = username
        self.isModerator = false
        self.isBanned = false
    }
}

/// Represents a message in the chatroom.
struct Message {
    let id: UUID
    let sender: User
    let content: String
    let timestamp: Date
}

/// Represents a chatroom where users can send messages.
class Chatroom {
    private var users: [UUID: User] = [:]
    private var messages: [Message] = []
    
    /// Adds a new user to the chatroom.
    /// - Parameter username: The username of the new user.
    /// - Returns: The newly created user.
    func addUser(username: String) -> User {
        let user = User(username: username)
        users[user.id] = user
        return user
    }
    
    /// Changes the username of an existing user.
    /// - Parameters:
    ///   - user: The user whose username is to be changed.
    ///   - newUsername: The new username.
    func changeUsername(user: User, newUsername: String) {
        user.username = newUsername
    }
    
    /// Adds a flair to a user.
    /// - Parameters:
    ///   - user: The user to whom the flair is to be added.
    ///   - flair: The flair to be added.
    func addFlair(user: User, flair: String) {
        user.flair = flair
    }
    
    /// Promotes a user to a moderator.
    /// - Parameter user: The user to be promoted.
    func promoteToModerator(user: User) {
        user.isModerator = true
        user.flair = "Mod"
    }
    
    /// Sends a message in the chatroom.
    /// - Parameters:
    ///   - sender: The user sending the message.
    ///   - content: The content of the message.
    func sendMessage(sender: User, content: String) {
        guard !sender.isBanned && sender.timeoutEndDate == nil else {
            return
        }
        
        let message = Message(id: UUID(), sender: sender, content: content, timestamp: Date())
        messages.append(message)
    }
    
    /// Bans a user from the chatroom.
    /// - Parameters:
    ///   - moderator: The moderator banning the user.
    ///   - userToBan: The user to be banned.
    func ban(moderator: User, userToBan: User) {
        guard moderator.isModerator else {
            return
        }
        
        userToBan.isBanned = true
    }
    
    /// Unbans a user from the chatroom.
    /// - Parameters:
    ///   - moderator: The moderator unbanning the user.
    ///   - userToUnban: The user to be unbanned.
    func unban(moderator: User, userToUnban: User)  {
        guard moderator.isModerator else {
            return
        }
        
        userToUnban.isBanned = false
    }
    
    /// Puts a user in timeout for a specified duration.
    /// - Parameters:
    ///   - moderator: The moderator putting the user in timeout.
    ///   - userToTimeout: The user to be put in timeout.
    ///   - duration: The duration of the timeout.
    func timeout(moderator: User, userToTimeout: User, duration: TimeInterval) {
        guard moderator.isModerator else {
            return
        }
        
        userToTimeout.timeoutEndDate = Date().addingTimeInterval(duration)
    }
    
    /// Ends the timeout for a user.
    /// - Parameters:
    ///   - moderator: The moderator ending the timeout.
    ///   - userToUntimeout: The user whose timeout is to be ended.
    func endTimeout(moderator: User, userToUntimeout: User) {
        guard moderator.isModerator else {
            return
        }
        
        userToUntimeout.timeoutEndDate = nil
    }
    
    /// Retrieves all messages in the chatroom.
    /// - Returns: An array of messages.
    func getMessages() -> [Message] {
        return messages
    }
}

let chatroom = Chatroom()

let alice = chatroom.addUser(username: "Alice")
let bob = chatroom.addUser(username: "Bob")
let charlie = chatroom.addUser(username: "Charlie")

chatroom.promoteToModerator(user: alice)
chatroom.addFlair(user: bob, flair: "Newbie")

chatroom.sendMessage(sender: alice, content: "Welcome to the chatroom!")
chatroom.sendMessage(sender: bob, content: "Hi everyone!")

chatroom.timeout(moderator: alice, userToTimeout: bob, duration: 3600) // 1 hour timeout

chatroom.sendMessage(sender: bob, content: "Can anyone see this?") // This will fail due to timeout

chatroom.changeUsername(user: charlie, newUsername: "Charles")

chatroom.sendMessage(sender: charlie, content: "Hi!") 

chatroom.sendMessage(sender: charlie, content: "Hi!") 

chatroom.endTimeout(moderator: alice, userToUntimeout: bob) // End Bob's timeout

chatroom.sendMessage(sender: bob, content: "Am I banned?")

for message in chatroom.getMessages() {
    if let flair = message.sender.flair{
    print("\(message.sender.username) [\(flair)]: \(message.content)")
    } else{
        print("\(message.sender.username): \(message.content)")
    }
}