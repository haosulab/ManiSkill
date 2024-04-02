# ManiSkill Documentation

Install Sphinx and Theme

```bash
# In the project root
pip install -e .[docs]
```

Build the documentation

```bash
# In docs/
make html
```

Start a server to watch changes

```bash
# In docs/
sphinx-autobuild ./source ./build/html
```


For github links for the time being must double check they link the right branch/commit