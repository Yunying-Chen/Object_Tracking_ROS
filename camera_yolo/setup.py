from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
# fetch values from package.xml
setup_args = generate_distutils_setup(
name = "camera_yolo",
packages=['camera_yolo'],
package_dir={'': 'src'},
)
setup(**setup_args)
