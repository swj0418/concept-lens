# concept-lens
Concept Lens repository for TVCG submission

# Workflow
## Step 1. Data Generation
We support 4 basic direction-finding methods applicable to style-based generative models.\
The supported methods are SeFA, GANSpace, Vector Arithmetic, and Supervised (SVM). However, you are free to explore
data prepared in the experiment data format.

Data generation is performed in 2 stages. The first stage is image generation and the second is the image processing.\
The default image generation scripts are prepared in the root. Simply run

```
sh {script}.sh
```

After the images are generated, you have to process the generated images to extract their features. The image processing\
module expects folder in the following structure, and you can bring your own generated data to inspect using ConceptLens.

### Data format for image processing
```
experiement_name/
├── codes/
│   ├── {image_idx}.jpg
│   └── …
└── walked/
    ├── {image_idx}-{direction_idx}.jpg
    └── …
```


## Step 2. Server-Client Setup

### 1. Server setup
Data server is implemented in Django.

Server-related dependencies
```
pip install django
pip install django-cors-headers
```

**Running the server**\
Communication port is defaulted to 37203.
```
cd server
python manage.py runserver 37203
```

### 2. Client setup
Client is implemented in React.\
When you start the client it will read the generated experiments from [Step 1] prepared in the resources folder.

**Running the client**\
Simply boot the client web-app and explore the data you generated.
```
cd client
npm install
npm start
```

## Step 3. Data Exploration
