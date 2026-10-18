# Contributing to ML4T Models

Follow the [shared ML4T contribution policy](https://github.com/ml4t/ecosystem/blob/main/CONTRIBUTING.md).
Search existing issues first. Describe one problem or coherent change per issue and submit one
focused PR linked to its owning issue. Automated dependency updates and administrative metadata
changes retain the shared exception.

Maintainer-directed work may group related issues in one bounded PR under the shared policy.
Keep changes together when they serve the same workflow or subsystem repair and can be reviewed
and reverted together. Split unrelated changes and changes needing independent release decisions.

For defects, provide a minimal reproduction through the public API or user workflow, package and
environment details, and expected behavior. Record a regression failing on the relevant base and
passing on the fix, with exact revisions, commands, and results. Correct verified numerical defects;
agree on intentional changes to definitions, annualization, signs, or defaults in the owning issue
and describe their compatibility impact. Documentation changes use relevant documentation checks.

Disclose whether AI assisted with code or prose and identify its contribution. Review and test your
submission yourself, and be prepared to explain its implementation, evidence, and review responses.

Preserve temporal identities, model-family assumptions, and optional dependency boundaries.
Retain the existing CPU/GPU qualification requirements for affected model families.
Use the repository's [quality commands](AGENTS.md#quality-commands) and report their outcomes.
Report suspected vulnerabilities through
[private security reporting](https://github.com/ml4t/models/security/advisories/new).
