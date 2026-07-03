import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abril/Documents/PrincipiosRobotica/Prueba/TP_final_robotica/ros_ws/install/algoritmos'
