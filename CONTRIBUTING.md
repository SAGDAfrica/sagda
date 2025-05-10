# Contributing to **SAGDA**

Thank you for considering contributing to **SAGDA**! We are thrilled to have you here, and we appreciate your efforts in improving this open-source project. Whether you're fixing bugs, adding features, improving documentation, or suggesting enhancements, your contributions are highly valued.

---

## **Table of Contents**

1. [Getting Started](#getting-started)
2. [How to Contribute](#how-to-contribute)

   * Reporting Bugs
   * Suggesting Features
   * Code Contributions
3. [Development Setup](#development-setup)
4. [Code Style and Guidelines](#code-style-and-guidelines)
5. [Pull Request Guidelines](#pull-request-guidelines)
6. [Testing Your Changes](#testing-your-changes)
7. [Code of Conduct](#code-of-conduct)
8. [Community Support](#community-support)
9. [License](#license)

---

## **Getting Started**

1. **Fork** the repository on GitHub.
2. **Clone** your fork:

   ```bash
   git clone https://github.com/YOUR-USERNAME/sagda.git
   cd sagda
   ```
3. **Add the original repository as a remote**:

   ```bash
   git remote add upstream https://github.com/SAGDAfrica/sagda.git
   ```

---

## **How to Contribute**

We welcome contributions of all types:

* **Bug Fixes**
* **Feature Enhancements**
* **Documentation Improvements**
* **Test Coverage Improvements**

---

### **Reporting Bugs**

If you discover a bug, please [open an issue](https://github.com/SAGDAfrica/sagda/issues) and include:

* A clear and descriptive title.
* Steps to reproduce the issue.
* Expected vs. actual behavior.
* Screenshots, error messages, or logs, if applicable.

---

### **Suggesting Features**

To propose a new feature:

* [Open a feature request](https://github.com/SAGDAfrica/sagda/issues) with:

  * A clear and descriptive title.
  * A detailed explanation of the feature and its use case.
  * If possible, suggest a potential implementation.

---

### **Code Contributions**

1. **Create a new branch** for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:

   * Ensure your code is well-structured and follows [PEP8 guidelines](https://www.python.org/dev/peps/pep-0008/).
   * Keep code modular and maintainable.
   * Document your code properly.

3. **Commit your changes**:

   ```bash
   git add .
   git commit -m "feat: Add feature description"
   ```

4. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** to the `main` branch.

---

##  **Development Setup**

1. Ensure you have Python 3.8+ installed.

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run tests to verify everything is working:

   ```bash
   pytest tests/
   ```

4. For development:

   ```bash
   pip install -e .
   ```

---

## **Code Style and Guidelines**

We follow:

* **PEP8** for Python code styling.
* **Google Docstrings** for documentation.
* **Black** for code formatting:

  ```bash
  black sagda/
  ```

---

## **Pull Request Guidelines**

To ensure smooth integration:

1. Ensure your PR follows the project's coding standards.
2. Reference any related issues (e.g., `Fixes #123`).
3. Ensure all tests pass before submitting.
4. Describe your changes clearly and concisely.
5. If adding a new feature, update the documentation accordingly.
6. Label your PR with appropriate tags (e.g., `bug`, `enhancement`, `documentation`).

---

## **Testing Your Changes**

All contributions must pass the testing suite. To run tests:

```bash
pytest tests/
```

* If you are adding a new feature, please write appropriate tests.
* If you are fixing a bug, include a test that prevents regression.

---

## **Code of Conduct**

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). We expect all contributors to adhere to it.

---

## **Community Support**

* Visit [sagda.org](https://sagda.org) for community discussions and resources.
* Engage with us on [GitHub Discussions](https://github.com/SAGDAfrica/sagda/discussions).
* For specific questions, feel free to [reach out](mailto:abdelghani.belgaid@um6p.ma).

---

## **License**

By contributing to SAGDA, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## **Thank You for Contributing!**

Your contribution helps **SAGDA** empower agricultural research across Africa. Together, we can drive innovation and sustainability in agricultural data science.
