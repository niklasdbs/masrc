from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
#module = Extension("envs.observation_creators.resource_observations.basic_resource_observation", sources=["envs.observation_creators.resource_observations.basic_resource_observation.pyx"])

setup(
        name='mtop',
        version='',
        packages=['envs', 'envs.observation_creators', 'envs.observation_creators.resource_observations',
                  'envs.observation_creators.resource_observations.resource_encoders', 'utils', 'trainers'],
        package_dir={'': 'src'},
        url='',
        license='',
        author='strauss',
        author_email='',
        description='',
        ext_modules=cythonize([  "src/envs/observation_creators/*.pyx",
                               "src/envs/observation_creators/resource_observations/resource_encoders/*.pyx",
                                "src/envs/observation_creators/resource_observations/basic_resource_observation.pyx",
                               ], compiler_directives={'language_level' : "3"}),
        zip_safe=False,
        include_dirs=[numpy.get_include()]
        #ext_modules=[module]
)
