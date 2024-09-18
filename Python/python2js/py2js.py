import frida
import sys

# Replace with your target package name
package_name = "com.your.package"

def on_message(message, data):
    if message['type'] == 'send':
        print("[*] {0}".format(message['payload']))
    else:
        print("[-] {0}".format(message['stacktrace']))

def main():
    try:
        # Attach to the running process
        process = frida.get_usb_device().attach(package_name)

        # JavaScript code to inject
        script_code = """
        Java.perform(function () {
            var targetClass = Java.use("%s.X.0bB"); // Replace with the full class name
            var classLoader = targetClass.class.getClassLoader();

            // Hook class loading
            var loadClass = classLoader.loadClass;
            classLoader.loadClass = function(className) {
                if (className === "%s.X.0bB") {
                    send("[+] Class loaded: " + className);

                    // Get constructor and its parameters
                    var constructors = targetClass.class.getDeclaredConstructors();
                    for (var i = 0; i < constructors.length; i++) {
                        send("[+] Constructor: " + constructors[i].toString());
                    }

                    // Get methods
                    var methods = targetClass.class.getDeclaredMethods();
                    for (var i = 0; i < methods.length; i++) {
                        send("[+] Method: " + methods[i].toString());
                    }
                }
                return loadClass.call(this, className);
            };
        });
        """ % (package_name, package_name)

        # Create and load the script
        script = process.create_script(script_code)
        script.on_message = on_message
        script.load()

        print("[*] Hooking class loading...")
        sys.stdin.read()  # Keep the script running

    except Exception as e:
        print(f"[-] An error occurred: {e}")

if __name__ == "__main__":
    main()