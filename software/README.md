# CoGNN

This folder contains the system prototype of `CoGNN` and other comparison methods. We run each method `10` times and present the average results to isolate the effects of randomness. 

## Environment

Add the path to this folder to `PYTHONPATH`.

## Usage

- Compile PyTorch for `CoGNN`, `PipeSwitch`, `MPS`, or `Default`. `CoGNN` and `PipeSwitch` could use the same modified PyTorch.

- For `CoGNN` and `PipeSwitch`, start the server first. After a few seconds, start the client to send requests.

- For `MPS`, enable the MPS server and run the execution script.

- For `Default`, directly run the execution script.

More details are included in README under folders for each system.
