# ManiSkill Documentation

Install Sphinx and Theme

```bash
# In the project root
pip install -e .[docs]
```

Start a server to watch changes

```bash
# In docs/
rm -rf build/ && sphinx-autobuild --ignore ./source/api ./source ./build/html
```

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