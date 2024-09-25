import Foundation

/// Represents a buff type that a skill can provide
enum BuffType {
    case strength
    case speed
    case defense
}

/// Represents a skill in the skill tree
struct Skill: Hashable {
    let id: UUID
    let name: String
    let buffType: BuffType
    let buffValue: Int
    
    init(name: String, buffType: BuffType, buffValue: Int) {
        self.id = UUID()
        self.name = name
        self.buffType = buffType
        self.buffValue = buffValue
    }
}

/// Manages the entire skill tree
struct SkillTree {
    private var skills: [Skill: Set<Skill>] = [:]
    private var unlockedSkills: Set<Skill> = []
    
    /// Adds a skill to the skill tree
    mutating func addSkill(_ skill: Skill, connectedTo parentSkills: [Skill] = []) {
        skills[skill] = Set(parentSkills)
        for parentSkill in parentSkills {
            skills[parentSkill, default: []].insert(skill)
        }
    }
    
    /// Attempts to unlock a skill by its name
    mutating func unlockSkill(named skillName: String) {
        guard let skill = skills.keys.first(where: { $0.name == skillName }) else {
            return 
        }
            unlockedSkills.insert(skill)
            }
    
    /// Retrieves all unlocked skills
    func getUnlockedSkills() -> [Skill] {
        return Array(unlockedSkills)
    }
    
    /// Calculates the total buff for a specific buff type
    func totalBuff(for buffType: BuffType) -> Int {
        return unlockedSkills
            .filter { $0.buffType == buffType }
            .reduce(0) { $0 + $1.buffValue }
    }
}

// Usage example
var skillTree = SkillTree()

let basicStrength = Skill(name: "Basic Strength", buffType: .strength, buffValue: 5)
let advancedStrength = Skill(name: "Advanced Strength", buffType: .strength, buffValue: 10)
let basicSpeed = Skill(name: "Basic Speed", buffType: .speed, buffValue: 5)

skillTree.addSkill(basicStrength)
skillTree.addSkill(advancedStrength, connectedTo: [basicStrength])
skillTree.addSkill(basicSpeed, connectedTo: [basicStrength])

// Unlock skills
skillTree.unlockSkill(named: "Basic Strength")
skillTree.unlockSkill(named: "Basic Speed")

// Calculate total strength buff
let totalStrengthBuff = skillTree.totalBuff(for: .strength)
print("Total Strength Buff: \(totalStrengthBuff)")

// Calculate total speed buff
let totalSpeedBuff = skillTree.totalBuff(for: .speed)
print("Total Speed Buff: \(totalSpeedBuff)")