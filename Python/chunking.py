import bpy

def create_mesh_from_chunk(chunk_data, name):
  """Creates a Blender mesh object from a terrain chunk.

  Args:
    chunk_data: 2D numpy array of elevation values.
    name: The name of the mesh object.
  """
  verts = []
  faces = []
  rows, cols = chunk_data.shape
  for i in range(rows):
    for j in range(cols):
      verts.append((i, j, chunk_data[i, j]))
      if i < rows - 1 and j < cols - 1:
        faces.append((i * cols + j, i * cols + j + 1, (i + 1) * cols + j + 1))
        faces.append((i * cols + j, (i + 1) * cols + j + 1, (i + 1) * cols + j))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    object = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(object)
    print(object)
    