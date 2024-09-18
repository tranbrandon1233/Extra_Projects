using UnityEngine;
using System.Collections.Generic;

public static class GameObjectSerialization
{
    // Serializable class to store component data
    [System.Serializable]
    public class SerializableComponent
    {
        public string type;
        public string data;

        public SerializableComponent(Component component)
        {
            type = component.GetType().AssemblyQualifiedName;
            data = JsonUtility.ToJson(component);
        }

        public Component Deserialize()
        {
            System.Type componentType = System.Type.GetType(type);
            Component component = gameObject.AddComponent(componentType);
            JsonUtility.FromJsonOverwrite(data, component);
            return component;
        }
    }

    // Serializable class to store GameObject data
    [System.Serializable]
    public class SerializableGameObject
    {
        public string name;
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 scale;
        public List<SerializableComponent> components;
        public List<SerializableGameObject> children;

        public SerializableGameObject(GameObject gameObject)
        {
            name = gameObject.name;
            position = gameObject.transform.position;
            rotation = gameObject.transform.rotation;
            scale = gameObject.transform.localScale;

            components = new List<SerializableComponent>();
            foreach (Component component in gameObject.GetComponents<Component>())
            {
                // Optionally exclude specific components here
                if (component is Transform) continue; 
                components.Add(new SerializableComponent(component));
            }

            children = new List<SerializableGameObject>();
            foreach (Transform child in gameObject.transform)
            {
                children.Add(new SerializableGameObject(child.gameObject));
            }
        }

        public GameObject Deserialize()
        {
            GameObject gameObject = new GameObject(name);
            gameObject.transform.position = position;
            gameObject.transform.rotation = rotation;
            gameObject.transform.localScale = scale;

            foreach (SerializableComponent serializedComponent in components)
            {
                serializedComponent.Deserialize();
            }

            foreach (SerializableGameObject serializedChild in children)
            {
                GameObject child = serializedChild.Deserialize();
                child.transform.SetParent(gameObject.transform);
            }

            return gameObject;
        }
    }

    // Serialize a GameObject to JSON string
    public static string Serialize(GameObject gameObject)
    {
        SerializableGameObject serializableGameObject = new SerializableGameObject(gameObject);
        return JsonUtility.ToJson(serializableGameObject, true);
    }

    // Deserialize a GameObject from a JSON string
    public static GameObject Deserialize(string json)
    {
        SerializableGameObject serializableGameObject = JsonUtility.FromJson<SerializableGameObject>(json);
        return serializableGameObject.Deserialize();
    }
}