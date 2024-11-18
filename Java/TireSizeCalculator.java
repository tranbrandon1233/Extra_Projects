public class TireSizeCalculator {

    /**
     * This method calculates the ideal tire width (in mm) based on the vehicle weight and horsepower,
     * as well as whether the car is for street or track use.
     *
     * The formula is influenced by the article 'How to Properly Select and Size Tires for Performance',
     * which emphasizes the importance of balancing vehicle weight and horsepower when selecting tire width.
     * Additionally, a minimum width of 195mm is applied based on common industry standards for
     * most performance cars, ensuring a practical lower limit for safety and functionality.
     *
     * @param vehicleWeight in kilograms
     * @param horsepower of the car
     * @param isTrack if true, it's for track use; if false, it's for street use
     * @return recommended tire width in millimeters
     */
    public static int calculateTireWidth(double vehicleWeight, int horsepower, boolean isTrack) {
        // Base tire width is roughly 1/10th of the vehicle weight
        double baseWidth = vehicleWeight / 10;
        
        // Adjust tire width based on horsepower
        double powerFactor = horsepower / 100.0;
        double widthAdjustment = baseWidth * (powerFactor / 2);  // higher horsepower cars need wider tires
        
        // Track use demands wider tires for improved grip and handling
        if (isTrack) {
            widthAdjustment *= 1.2; // 20% wider for track use
        }

        // Calculate the final width and enforce a minimum of 195mm, as commonly used in performance cars
        int recommendedWidth = (int) (baseWidth + widthAdjustment);
        return Math.max(recommendedWidth, 195); // Minimum width of 195mm
    }

    /**
     * This method provides a detailed tire recommendation based on vehicle specs and intended use.
     * It returns the tire width and a brief suggestion on tire type.
     */
    public static String getTireRecommendation(double vehicleWeight, int horsepower, boolean isTrack) {
        int width = calculateTireWidth(vehicleWeight, horsepower, isTrack);
        String useType = isTrack ? "track" : "street";
        return "Recommended tire width for " + useType + " use: " + width + " mm. "
             + (isTrack ? "Consider using slick or semi-slick tires for maximum grip on track." 
                        : "For street use, all-season or performance tires will be suitable.");
    }

    public static void main(String[] args) {
        // Example usage
        double vehicleWeight = 1400.0;  // weight in kg
        int horsepower = 300;           // horsepower
        boolean isTrack = true;         // track or street
        
        // Print tire recommendation
        System.out.println(getTireRecommendation(vehicleWeight, horsepower, isTrack));
    }
}
