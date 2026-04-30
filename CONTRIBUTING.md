# Contributing

Thanks for considering contributing to this project! This project is forked from [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (Apache 2.0 License) and customized for U0 underwater robot applications. Please read this document to learn how to contribute.

## Bug Reports and Feature Requests

### Did you find a bug?

First, [search existing issues](https://gitee.com/vincent-gu/u0/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, [open a new issue](https://gitee.com/vincent-gu/u0/issues/new). Be sure to include a clear title and description. The description should include as much relevant information as possible, explain how to reproduce the erroneous behavior as well as the behavior you expect to see. Ideally you would include a code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use Gitee issues to track feature requests. Before you create a feature request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing it first in an issue.
* Check the documentation to make sure your feature does not already exist.
* [Search existing issues](https://gitee.com/vincent-gu/u0/issues) to see whether your feature has already been suggested.

When creating your request, please:

* Provide a clear title and description.
* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
* Include code examples to demonstrate how the enhancement would be used.

## Making a Pull Request

When you're ready to contribute code to address an open issue, please follow these guidelines.

1. **Initial setup** (only do this once)

    <details><summary>Expand details</summary><br/>

    If you haven't already done so, please [fork](https://gitee.com/vincent-gu/u0/fork) this repository on Gitee.

    Then clone your fork locally with

        git clone https://gitee.com/USERNAME/u0.git

    or

        git clone git@gitee.com:USERNAME/u0.git

    Create a Python 3 virtual environment. We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

        conda create -n gr00t python=3.10
        conda activate gr00t

    Once your virtual environment is activated, install your local clone in "editable mode" with

        pip install -U pip setuptools wheel
        pip install -e .[base]
        pip install --no-build-isolation flash-attn==2.7.1.post4

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details</summary><br/>

    Keep your fork up-to-date with the main repository:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Test your changes**

    <details><summary>Expand details</summary><br/>

    Before opening a PR, please make sure your changes pass the existing tests. You can run tests locally with [pytest](https://docs.pytest.org/en/latest/):

        pytest -v tests/

    You should also run code formatters to ensure consistent style:

        isort .
        black .

    And lint the code:

        ruff check . --fix

    After all checks have passed, you can [open a new Pull Request](https://gitee.com/vincent-gu/u0/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    </details>

## License

By contributing to this project, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), consistent with the upstream [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) project.

### Developer Certificate of Origin

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
