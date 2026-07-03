colcon build --packages-select parteb
source install/setup.bash
ros2 launch parteb parteb.launch.py map_path:=/RUTA/A/WS/DONDE/ESTÁ/mapa.npy robot:=simulado
```
# TERMINAL 1: Simulación en la casa personalizada
ros2 launch turtlebot3_custom_simulation custom_casa.launch.py

# TERMINAL 2: Control teleoperado
ros2 run turtlebot3_teleop teleop_keyboard

# TERMINAL 3: Build y ejecución de Fast SLAM
colcon build --symlink-install
source install/setup.bash
ros2 run partea slam_node

# TERMINAL 4: Interfaz gráfica
rviz2
    -> en rviz: add -> by topic -> map

# Hay que ver si se guarda bien el mapa y cómo: ros2 run partea map_saver_node
