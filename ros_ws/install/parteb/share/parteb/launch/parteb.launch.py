from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value='/home/alumno1/Documents/ros_ws/mapa.npz',
        description='Ruta al archivo .npy del mapa',
    )
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='simulado',
        description='Tipo de robot: simulado o real',
    )
    rviz_dir_arg = DeclareLaunchArgument(
        'rviz_dir',
        default_value='/home/alumno1/Documents/ros_ws/src/rviz',
        description='Directorio con los .rviz',
    )

    map_path = LaunchConfiguration('map_path')
    robot = LaunchConfiguration('robot')
    rviz_dir = LaunchConfiguration('rviz_dir')

    nodo_b = Node(
        package='parteb',
        executable='nodo_b',
        name='nodo_b',
        output='screen',
        parameters=[{'map_path': map_path, 'robot': robot}],
    )

    rviz_sim = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [rviz_dir, '/rviz2_config_B.rviz']],
        condition=IfCondition(PythonExpression(["'", robot, "' == 'simulado'"])),
    )

    rviz_real = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [rviz_dir, '/rviz2_config_B_real.rviz']],
        condition=IfCondition(PythonExpression(["'", robot, "' == 'real'"])),
    )

    return LaunchDescription([
        map_path_arg,
        robot_arg,
        rviz_dir_arg,
        nodo_b,
        rviz_sim,
        rviz_real,
    ])