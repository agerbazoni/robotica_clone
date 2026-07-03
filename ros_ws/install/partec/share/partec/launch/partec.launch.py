import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Configs de RViz ubicadas junto a este launch file. La real usa los tópicos
    # prefijados con /tb4_0; la de sim usa los tópicos sin prefijo.
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    rviz_config_sim = os.path.join(launch_dir, 'rviz.rviz')
    rviz_config_real = os.path.join(launch_dir, 'rviz_real.rviz')

    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='simulado',
        description="Tipo de robot: 'simulado' o 'real'",
    )
    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value='/home/alumno1/Documents/ros_ws/mapa.npz',
        description='Ruta al archivo .npz del mapa (usado por nodo_b de parteb)',
    )
    camera_topic_sim_arg = DeclareLaunchArgument(
        'camera_topic_sim',
        default_value='/camera/image_raw',
        description='Tópico de imagen a usar cuando robot:=simulado (ajustar según el mundo de Gazebo)',
    )
    camera_info_topic_sim_arg = DeclareLaunchArgument(
        'camera_info_topic_sim',
        default_value='/camera/camera_info',
        description='Tópico de camera_info a usar cuando robot:=simulado',
    )
    use_nodo_b_arg = DeclareLaunchArgument(
        'use_nodo_b',
        default_value='true',
        description='Si es true, levanta también nodo_b (parteb) para ejecutar localización/planificación/control',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Si es true, levanta RViz con la config rviz.rviz de este directorio',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Usar el reloj de /clock (poner en true al reproducir un rosbag con "ros2 bag play --clock")',
    )

    robot = LaunchConfiguration('robot')
    map_path = LaunchConfiguration('map_path')
    camera_topic_sim = LaunchConfiguration('camera_topic_sim')
    camera_info_topic_sim = LaunchConfiguration('camera_info_topic_sim')
    use_nodo_b = LaunchConfiguration('use_nodo_b')
    use_rviz = LaunchConfiguration('use_rviz')
    # Casteado a bool: use_sim_time es un parámetro booleano de rclpy.
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    vision_node = Node(
        package='partec',
        executable='vision',
        name='vision_node',
        output='screen',
        parameters=[{
            'robot': robot,
            'camera_topic_sim': camera_topic_sim,
            'camera_info_topic_sim': camera_info_topic_sim,
            'use_sim_time': use_sim_time,
        }],
    )

    cerebro_node = Node(
        package='partec',
        executable='cerebro',
        name='state_machine_node',
        output='screen',
        parameters=[{'robot': robot, 'map_path': map_path, 'use_sim_time': use_sim_time}],
    )

    # nodo_b de parteb resuelve localización + planificación + path following;
    # el cerebro de partec solo decide A DONDE mandarlo (explorar vs. ir al cono).
    nodo_b_node = Node(
        package='parteb',
        executable='nodo_b',
        name='nodo_b',
        output='screen',
        parameters=[{'map_path': map_path, 'robot': robot, 'use_sim_time': use_sim_time}],
        condition=IfCondition(use_nodo_b),
    )

    rviz_sim = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_sim],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression(
            ["'", use_rviz, "' == 'true' and '", robot, "' == 'simulado'"])),
    )

    rviz_real = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_real],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression(
            ["'", use_rviz, "' == 'true' and '", robot, "' == 'real'"])),
    )

    return LaunchDescription([
        robot_arg,
        map_path_arg,
        camera_topic_sim_arg,
        camera_info_topic_sim_arg,
        use_nodo_b_arg,
        use_rviz_arg,
        use_sim_time_arg,
        vision_node,
        cerebro_node,
        nodo_b_node,
        rviz_sim,
        rviz_real,
    ])