# sionna_tr38811
implementing the functionalities of 3gpp TR38.811 in the sionna framework

The implementation is based on the open source implementation of Sionnas channel according to the tr38901 standard. The structure is
reused, the calculations and necessary parameters are adapted.

This package needs to be added to the "sionna/channel" directory, this is the same level as tr38901. 
The environment setup is the regular Sionna setup. An example is found in sionna_tr38811.yaml.
As only the tr38811 directory is adapted everything else is in gitignore. The utils.py on channel level needs to be adapated though, 
as it asserts only for the tr38901 scenarios out of the box. 
