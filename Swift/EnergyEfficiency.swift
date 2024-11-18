import Foundation

struct EnergyMeasure {
    let name: String
    let baselineUsage: Double
    let estimatedReduction: Double
    let seasonalAdjustment: Double
    
    func calculateSeasonalSavings() -> Double {
        return baselineUsage * estimatedReduction * seasonalAdjustment
    }
}

struct BuildingProgram {
    let buildingName: String
    let measures: [EnergyMeasure]
    
    func calculateTotalSavings() -> Double {
        return measures.reduce(0) { $0 + $1.calculateSeasonalSavings() }
    }
    
    func generateDetailedReport() -> String {
        var report = "Building: \(buildingName)\n"
        measures.forEach { measure in
            report += "\(measure.name): \(measure.calculateSeasonalSavings()) kWh saved\n"
        }
        report += "Total Savings: \(calculateTotalSavings()) kWh"
        return report
    }
}

struct EnergyEfficiencyProgram {
    let buildingPrograms: [BuildingProgram]
    
    func totalProgramSavings() -> Double {
        return buildingPrograms.reduce(0) { $0 + $1.calculateTotalSavings() }
    }
    
    func generateOverallReport() -> String {
        var report = "Energy Efficiency Program Report\n"
        buildingPrograms.forEach { program in
            report += "\n" + program.generateDetailedReport() + "\n"
        }
        report += "\nOverall Total Savings: \(totalProgramSavings()) kWh"
        return report
    }
    
    func prioritizeMeasures() -> [EnergyMeasure] {
        return buildingPrograms
            .flatMap { $0.measures }
            .sorted { $0.calculateSeasonalSavings() > $1.calculateSeasonalSavings() }
    }
    
    func recommendTopMeasures(limit: Int) -> String {
        let topMeasures = prioritizeMeasures().prefix(limit)
        return topMeasures.enumerated()
            .map { "\($0 + 1). \($1.name): \($1.calculateSeasonalSavings()) kWh saved" }
            .joined(separator: "\n")
    }
}

func runValidations() {
    // Test 1: Calculate Seasonal Savings
    do {
        let measure = EnergyMeasure(name: "LED Lighting", baselineUsage: 1000, estimatedReduction: 0.2, seasonalAdjustment: 1.0)
        let savings = measure.calculateSeasonalSavings()
        assert(abs(savings - 200.0) < 0.001, "Expected 200.0, got \(savings)")
        print("✅ Test 1 passed: Calculate Seasonal Savings")
    }
    
    // Test 2: Calculate Seasonal Savings with Adjustment
    do {
        let measure = EnergyMeasure(name: "HVAC Upgrade", baselineUsage: 2000, estimatedReduction: 0.15, seasonalAdjustment: 0.8)
        let savings = measure.calculateSeasonalSavings()
        assert(abs(savings - 240.0) < 0.001, "Expected 240.0, got \(savings)")
        print("✅ Test 2 passed: Calculate Seasonal Savings with Adjustment")
    }
    
    // Test 3: Calculate Total Savings
    do {
        let measures = [
            EnergyMeasure(name: "LED Lighting", baselineUsage: 1000, estimatedReduction: 0.2, seasonalAdjustment: 1.0),
            EnergyMeasure(name: "HVAC Upgrade", baselineUsage: 2000, estimatedReduction: 0.15, seasonalAdjustment: 0.8)
        ]
        let buildingProgram = BuildingProgram(buildingName: "City Hall", measures: measures)
        let totalSavings = buildingProgram.calculateTotalSavings()
        assert(abs(totalSavings - 440.0) < 0.001, "Expected 440.0, got \(totalSavings)")
        print("✅ Test 3 passed: Calculate Total Savings")
    }
    
    // Test 4: Generate Detailed Report
    do {
        let measures = [
            EnergyMeasure(name: "LED Lighting", baselineUsage: 1000, estimatedReduction: 0.2, seasonalAdjustment: 1.0),
            EnergyMeasure(name: "HVAC Upgrade", baselineUsage: 2000, estimatedReduction: 0.15, seasonalAdjustment: 0.8)
        ]
        let buildingProgram = BuildingProgram(buildingName: "City Hall", measures: measures)
        let report = buildingProgram.generateDetailedReport()
        assert(report.contains("LED Lighting: 200.0 kWh saved"))
        assert(report.contains("HVAC Upgrade: 240.0 kWh saved"))
        assert(report.contains("Total Savings: 440.0 kWh"))
        print("✅ Test 4 passed: Generate Detailed Report")
    }
    
    // Test 5: Total Program Savings
    do {
        let measures1 = [
            EnergyMeasure(name: "LED Lighting", baselineUsage: 1000, estimatedReduction: 0.2, seasonalAdjustment: 1.0)
        ]
        let measures2 = [
            EnergyMeasure(name: "HVAC Upgrade", baselineUsage: 2000, estimatedReduction: 0.15, seasonalAdjustment: 0.8)
        ]
        let buildingPrograms = [
            BuildingProgram(buildingName: "City Hall", measures: measures1),
            BuildingProgram(buildingName: "Library", measures: measures2)
        ]
        let program = EnergyEfficiencyProgram(buildingPrograms: buildingPrograms)
        let totalSavings = program.totalProgramSavings()
        assert(abs(totalSavings - 440.0) < 0.001, "Expected 440.0, got \(totalSavings)")
        print("✅ Test 5 passed: Total Program Savings")
    }
    
    print("\nAll validations completed successfully! 🎉")
}

// Run the validations
runValidations()