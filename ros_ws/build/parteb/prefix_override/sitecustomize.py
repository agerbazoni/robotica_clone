import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lucas/Documents/UDESA/7cuatrimestre/Principios_de_la_Robotica_Autonoma/TPs/TP_final_robotica/ros_ws/install/parteb'
