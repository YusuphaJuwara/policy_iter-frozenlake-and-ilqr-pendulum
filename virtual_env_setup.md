# Python Virtual Environment (venv) Setup and Management

This guide explains how to create, activate, and manage Python virtual environments (venv) to keep your project dependencies isolated and organized.

## Table of Contents

1. [Creating a Virtual Environment](#creating-a-virtual-environment)
2. [Activating a Virtual Environment](#activating-a-virtual-environment)
3. [Deactivating a Virtual Environment](#deactivating-a-virtual-environment)
4. [Installing Libraries](#installing-libraries)
5. [Uninstalling Libraries](#uninstalling-libraries)
6. [System-Wide Libraries vs. Virtual Environment Libraries](#system-wide-libraries-vs-virtual-environment-libraries)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Creating a Virtual Environment

To create a new virtual environment, open your terminal and navigate to your project's root directory. Run the following command to create a venv:

```bash
python -m venv hm1venv
```

Replace `hm1venv` with the desired name for your virtual environment. This command will create a new directory with the chosen name, containing a fresh Python environment.

To use a specific python version, in this case version `3.8` to create a virtual environment named `a3c3_38_32`:

```bash
C:\Users\yusup\AppData\Local\Programs\Python\Python38-32\python.exe -m venv a3c3_38_32
```

## Activating a Virtual Environment

Before using your virtual environment, you need to activate it. Use the appropriate command based on your operating system:

- On Windows:

```bash
myenv\Scripts\activate
```

- On macOS/Linux:

```bash
source myenv/bin/activate
```

Replace `myenv` with your virtual environment name. Once activated, your terminal prompt will indicate the active virtual environment.

## Deactivating a Virtual Environment

To deactivate your virtual environment, simply run:

```bash
deactivate
```

This will return you to the system-wide Python environment.

## Installing Libraries

With your virtual environment activated, you can install Python libraries specific to your project. Use `pip` to install libraries:

```bash
pip install library-name
```

Replace `library-name` with the desired library. Libraries installed in this manner are isolated to your virtual environment.

To install all the libraries in a requirements.txt file, use the following command

```bash
pip install -r requirements.txt
```

## Uninstalling Libraries

To uninstall a library from your virtual environment, use the following command:

```bash
pip uninstall library-name
```

Replace `library-name` with the library you want to uninstall.

## System-Wide Libraries vs. Virtual Environment Libraries

- If a library is installed system-wide (outside the virtual environment), it won't be available within the venv unless you install it separately in the virtual environment.
- Libraries installed in a virtual environment do not affect your system-wide Python environment.

## Best Practices

- Create a dedicated virtual environment for each project to avoid conflicts between project dependencies.
- Always activate your virtual environment before running your project or installing libraries.
- Include your virtual environment directory (e.g., `myenv/`) in your project's `.gitignore` file to prevent version control of the environment.

## Troubleshooting

- If you forget to deactivate your virtual environment and close the terminal, simply reactivate the environment using the activation command.
- If you encounter library conflicts or issues, create a new virtual environment for your project.

## Check versions

To check all the installed library versions, run the following command

```bash
python -m pip freeze
```

## Copy versions into a requiremenst.txt file

- `pip freeze`: Lists all installed packages and their versions.
- `>`: Redirects the output to a file.
- `requirements.txt`: The file where the package information will be stored.

```bash
pip freeze > requirements.txt
```
