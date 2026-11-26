# Known faces

Create one directory per person and place one or more clear, front-facing images
inside it:

```text
known_faces/
|-- Ada Lovelace/
|   |-- portrait-1.jpg
|   `-- portrait-2.jpg
`-- Grace Hopper/
    `-- portrait.jpg
```

The directory name becomes the displayed identity. Images should contain one
face. Locally enrolled profiles are ignored by Git because face images and
embeddings are biometric data; only the fictional `Sample Person` fixture is
allowlisted for the tutorial smoke test.
