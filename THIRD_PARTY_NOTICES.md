# Third-Party Notices

This repository implements a C inference runtime compatible with VoxCPM2 and
uses ggml as its tensor backend. Model weights are not included in source or
release packages.

## OpenBMB VoxCPM / VoxCPM2

- Project: <https://github.com/OpenBMB/VoxCPM>
- License: Apache License 2.0
- License text: <https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE>

The upstream project states that its code and VoxCPM model weights are released
under Apache-2.0. Users who download weights remain responsible for checking the
license and terms attached to the exact model snapshot they use.

## ggml

- Project: <https://github.com/ggml-org/ggml>
- License: MIT
- License text: <https://github.com/ggml-org/ggml/blob/master/LICENSE>

ggml is resolved by CMake, normally through `FetchContent`, and is not copied
into this repository's source tree. The license shipped by the resolved ggml
revision is authoritative.

## Project License Status

This notice records third-party licensing only. It does not select or change the
license for original `voxcpm-cpp` code; that project-level licensing decision
must be made by the repository owner.
