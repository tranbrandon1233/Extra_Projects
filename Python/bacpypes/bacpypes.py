from bacpypes.core import run, stop
from bacpypes.pdu import Address
from bacpypes.app import BIPSimpleApplication
from bacpypes.local.device import LocalDeviceObject
from bacpypes.object import get_object_class, Property
from bacpypes.apdu import WritePropertyRequest, SimpleAckPDU

# BACnet device information
DEVICE_ID = 1234  # Replace with your device ID
OBJECT_TYPE = 'analogValue'  # Replace with the desired object type
OBJECT_INSTANCE = 1  # Replace with the desired object instance
NEW_DESCRIPTION = 'New Description'  # Replace with the new description

# BAC0 configuration
BAC0_INTERFACE = 'eth0'  # Replace with your BACnet interface

class WritePropertyApp(BIPSimpleApplication):
    def __init__(self, *args):
        BIPSimpleApplication.__init__(self, *args)

    def request(self, apdu):
        if isinstance(apdu, WritePropertyRequest):
            # Check if the request is for the description property
            if apdu.objectIdentifier[0] == get_object_class(OBJECT_TYPE) and \
               apdu.objectIdentifier[1] == OBJECT_INSTANCE and \
               apdu.propertyIdentifier == 'description':

                # Update the description property
                self.localDevice.objectList[0].properties['description'] = Property(
                    'description', NEW_DESCRIPTION, 28
                )

                # Send a SimpleAck response
                response = SimpleAckPDU(context=apdu)
                self.response(response)
            else:
                BIPSimpleApplication.request(self, apdu)
        else:
            BIPSimpleApplication.request(self, apdu)

def main():
    # Create a local device object
    local_device = LocalDeviceObject(
        objectName='MyDevice',
        objectIdentifier=('device', DEVICE_ID),
        maxApduLengthAccepted=1024,
        segmentationSupported='segmentedBoth',
        vendorID=0,
    )

    # Create an instance of the application
    app = WritePropertyApp(local_device, Address(f'{BAC0_INTERFACE}:47808'))

    # Start the BACnet stack
    run()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        stop()