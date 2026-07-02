from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
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
        description='Si es true, levanta RViz con la config de Parte B',
    )
    rviz_dir_arg = DeclareLaunchArgument(
        'rviz_dir',
        default_value='/home/alumno1/Documents/ros_ws/src/rviz',
        description='Directorio con los .rviz',
    )

    robot = LaunchConfiguration('robot')
    map_path = LaunchConfiguration('map_path')
    camera_topic_sim = LaunchConfiguration('camera_topic_sim')
    camera_info_topic_sim = LaunchConfiguration('camera_info_topic_sim')
    use_nodo_b = LaunchConfiguration('use_nodo_b')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_dir = LaunchConfiguration('rviz_dir')

    vision_node = Node(
        package='partec',
        executable='vision',
        name='vision_node',
        output='screen',
        parameters=[{
            'robot': robot,
            'camera_topic_sim': camera_topic_sim,
            'camera_info_topic_sim': camera_info_topic_sim,
        }],
    )

    cerebro_node = Node(
        package='partec',
        executable='cerebro',
        name='state_machine_node',
        output='screen',
        parameters=[{'robot': robot, 'map_path': map_path}],
    )

    # nodo_b de parteb resuelve localización + planificación + path following;
    # el cerebro de partec solo decide A DONDE mandarlo (explorar vs. ir al cono).
    nodo_b_node = Node(
        package='parteb',
        executable='nodo_b',
        name='nodo_b',
        output='screen',
        parameters=[{'map_path': map_path, 'robot': robot}],
        condition=IfCondition(use_nodo_b),
    )

    rviz_sim = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [rviz_dir, '/rviz2_config_B.rviz']],
        condition=IfCondition(PythonExpression(
            ["'", use_rviz, "' == 'true' and '", robot, "' == 'simulado'"])),
    )

    rviz_real = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [rviz_dir, '/rviz2_config_B_real.rviz']],
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
        rviz_dir_arg,
        vision_node,
        cerebro_node,
        nodo_b_node,
        rviz_sim,
        rviz_real,
    ])