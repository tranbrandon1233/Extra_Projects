import Foundation

struct CPUTask {
    let id: Int
    let cpuRequired: Int
    let efficiencyGain: Int
}

func findBestOrder(tasks: [CPUTask], maxCpu: Int) -> [[CPUTask]] {
    let sortedTasks = tasks.sorted { $0.efficiencyGain > $1.efficiencyGain }
    
    var cycles = [[CPUTask]]()
    var usedTaskIds = Set<Int>()
    
    while !sortedTasks.allSatisfy({ usedTaskIds.contains($0.id) }) {
        var currentCycle = [CPUTask]()
        var currentCpu = 0
        
        for task in sortedTasks {
            if !usedTaskIds.contains(task.id) && currentCpu + task.cpuRequired <= maxCpu {
                currentCycle.append(task)
                currentCpu += task.cpuRequired
                usedTaskIds.insert(task.id)
            }
        }
        
        if !currentCycle.isEmpty {
            cycles.append(currentCycle)
        }
    }
    
    return cycles
}

let tasks = [
    CPUTask(id: 1, cpuRequired: 1, efficiencyGain: 60),
    CPUTask(id: 2, cpuRequired: 1, efficiencyGain: 100),
    CPUTask(id: 3, cpuRequired: 1, efficiencyGain: 120),
    CPUTask(id: 4, cpuRequired: 1, efficiencyGain: 90)
]

let maxCpu = 60
let bestOrder = findBestOrder(tasks: tasks, maxCpu: maxCpu)

print("The max CPU processing load available is \(maxCpu).")

// Print the best order of tasks for each cycle
for (cycleIndex, cycle) in bestOrder.enumerated() {
    print("Cycle \(cycleIndex + 1):")
    for task in cycle {
        print("  Task ID: \(task.id), CPU Required: \(task.cpuRequired), Efficiency Gain: \(task.efficiencyGain)")
    }
}