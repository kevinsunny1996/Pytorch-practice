# Pytorch-practice
Credits - https://github.com/pytorch/examples

# Primary objective for using this repo for practice:
* To get familiar with pytorch syntax.
* To get to know the best practices while coding a DNN.
* The repo uses argparse library to create CLI style code.

# Secondary objectives for using this repo:
* To learn to debug pytorch via VSCode IDE debugger tool and not via notebook.
* To learn to setup Windows Subsystem for Linux in the new laptop and point VSCode to that disk image.
* To setup python ecosystem on WSL.

# Difficulties faced while setting up the workspace:
* Python linter was not able to detect few methods.
* Dimensions of the layers were passed wrongly and were throwing an error.

# Solutions to the above:
* Add the respective entries in the pylint settings (Linter that was used for this).
* Dimension issue was detected when running the debugger tool and it was easy to detect since it was happening during training.
* The stacktrace interpretation pointed it to the line where the matrix multiplication happened and the line was compared with the one in the repo and helped in fixing the issue.