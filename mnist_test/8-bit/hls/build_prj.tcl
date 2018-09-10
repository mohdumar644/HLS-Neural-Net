############################################################
# create HLS project
############################################################
open_project -reset mnist_deep_proj
set_top deepMNIST
add_files core.cpp
add_files -tb test_core.cpp
open_solution -reset "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
exit
