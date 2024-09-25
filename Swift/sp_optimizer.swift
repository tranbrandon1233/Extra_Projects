import Foundation

struct Point {
    let x: Double
    let y: Double
}

struct RoofSection {
    let topLeft: Point
    let width: Double
    let height: Double
    let orientation: Double // in degrees, 0 is South, 90 is West, etc.
    let tilt: Double // in degrees
}

struct ShadingObject {
    let position: Point
    let height: Double
    let radius: Double
}

struct SolarData {
    let hour: Int
    let directIrradiance: Double
    let diffuseIrradiance: Double
}

struct SolarPanel {
    let position: Point
    let width: Double
    let height: Double
}

class SolarPanelOptimizer {

    static let standardPanelWidth: Double = 1.0
    static let standardPanelHeight: Double = 1.6

    static func optimizePanelPlacement(roofSections: [RoofSection], shadingObjects: [ShadingObject], solarData: [SolarData]) -> [SolarPanel] {
        var optimalPanels: [SolarPanel] = []

        for section in roofSections {
            let sectionPanels = placePanelsInSection(section: section, shadingObjects: shadingObjects, solarData: solarData)
            optimalPanels.append(contentsOf: sectionPanels)
        }

        return optimalPanels
    }

    private static func placePanelsInSection(section: RoofSection, shadingObjects: [ShadingObject], solarData: [SolarData]) -> [SolarPanel] {
        var panels: [SolarPanel] = []

        let panelsPerRow = Int(section.width / standardPanelWidth)
        let panelsPerColumn = Int(section.height / standardPanelHeight)

        for row in 0..<panelsPerRow {
            for col in 0..<panelsPerColumn {
                let x = section.topLeft.x + Double(row) * standardPanelWidth
                let y = section.topLeft.y + Double(col) * standardPanelHeight
                let panelPosition = Point(x: x, y: y)

                if !isShaded(position: panelPosition, shadingObjects: shadingObjects, section: section, solarData: solarData) {
                    let panel = SolarPanel(position: panelPosition, width: standardPanelWidth, height: standardPanelHeight)
                    panels.append(panel)
                }
            }
        }

        return panels
    }

    private static func isShaded(position: Point, shadingObjects: [ShadingObject], section: RoofSection, solarData: [SolarData]) -> Bool {
        for object in shadingObjects {
            let distance = sqrt(pow(position.x - object.position.x, 2) + pow(position.y - object.position.y, 2))
            if distance < object.radius {
                return true
            }
        }

        return false
    }
}

let roofSections = [
    RoofSection(topLeft: Point(x: 0, y: 0), width: 10, height: 5, orientation: 180, tilt: 30),
    RoofSection(topLeft: Point(x: 0, y: 5), width: 8, height: 4, orientation: 90, tilt: 25)
]

let shadingObjects = [
    ShadingObject(position: Point(x: 5, y: 2), height: 2, radius: 1),
    ShadingObject(position: Point(x: 7, y: 7), height: 3, radius: 1.5)
]

let solarData = [
    SolarData(hour: 8, directIrradiance: 200, diffuseIrradiance: 50),
    SolarData(hour: 12, directIrradiance: 800, diffuseIrradiance: 100),
    SolarData(hour: 16, directIrradiance: 400, diffuseIrradiance: 75)
]

let optimalPanels = SolarPanelOptimizer.optimizePanelPlacement(roofSections: roofSections, shadingObjects: shadingObjects, solarData: solarData)

print("Optimal solar panel arrangement:")
for (index, panel) in optimalPanels.enumerated() {
    print("Panel \(index + 1): Position (x: \(panel.position.x), y: \(panel.position.y))")
}
print("Total number of panels: \(optimalPanels.count)")