# import ptvsd
import debugpy
import sys

def break_into_debugger_(port= 6789):
    ip_address = ('0.0.0.0', port)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
    # Allow other computers to attach to ptvsd at this IP address and port.
    ptvsd.enable_attach(address=ip_address)
    # Pause the program until a remote debugger is attached
    ptvsd.wait_for_attach()
    print("Process attached, start running into experiment...", flush= True)
    ptvsd.break_into_debugger()

def break_into_debugger(port= 6789):
    ip_address = ('0.0.0.0', port)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()
