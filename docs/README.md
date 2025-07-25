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

## Auto Generate Task Docs

```bash
# In docs/
python generate_task_docs.py
```

## Auto Generate Robot Docs

```bash
# In docs/

# generate all robot docs
python generate_robot_docs.py

# update just one robot's documentation
python generate_robot_docs.py robot_uid
```
