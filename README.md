colcon build --packages-select parteb
source install/setup.bash
ros2 launch parteb parteb.launch.py map_path:=/RUTA/A/WS/DONDE/ESTÁ/mapa.npy robot:=simulado