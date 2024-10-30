class RubiksCube {
    constructor() {
      this.cube = this.createSolvedCube();
    }
  
    createSolvedCube() {
      const cube = new Array(27);
      for (let i = 0; i < 27; i++) {
        cube[i] = Math.floor(i / 9); // Simple color assignment based on slice
      }
      return cube;
    }
  
    // Helper function to get coordinates from index
    getCoords(index) {
      const z = Math.floor(index / 9);
      const y = Math.floor((index % 9) / 3);
      const x = index % 3;
      return { x, y, z };
    }
  
    // Helper function to get index from coordinates
    getIndex(x, y, z) {
      return z * 9 + y * 3 + x;
    }
  
    rotateXSlice(sliceIndex) {
      const temp = [...this.cube]; // Create a copy to avoid modifying in place
      for (let y = 0; y < 3; y++) {
        for (let z = 0; z < 3; z++) {
          const oldIndex = this.getIndex(sliceIndex, y, z);
          const newIndex = this.getIndex(sliceIndex, 2 - z, y);
          this.cube[newIndex] = temp[oldIndex];
        }
      }
    }
  
    rotateYSlice(sliceIndex) {
      const temp = [...this.cube];
      for (let x = 0; x < 3; x++) {
        for (let z = 0; z < 3; z++) {
          const oldIndex = this.getIndex(x, sliceIndex, z);
          const newIndex = this.getIndex(z, sliceIndex, 2 - x);
          this.cube[newIndex] = temp[oldIndex];
        }
      }
    }
  
    rotateZSlice(sliceIndex) {
      const temp = [...this.cube];
      for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 3; y++) {
          const oldIndex = this.getIndex(x, y, sliceIndex);
          const newIndex = this.getIndex(2 - y, x, sliceIndex);
          this.cube[newIndex] = temp[oldIndex];
        }
      }
    }
  }
  
  // Example Usage:
  const cube = new RubiksCube();
  console.log("Initial Cube:", cube.cube);
  
  cube.rotateXSlice(0); // Rotate the first slice along the X-axis
  console.log("Cube after X-slice rotation:", cube.cube);
  
  cube.rotateYSlice(1); // Rotate the second slice along the Y-axis
  console.log("Cube after Y-slice rotation:", cube.cube);
  
  cube.rotateZSlice(2); // Rotate the third slice along the Z-axis
  console.log("Cube after Z-slice rotation:", cube.cube);