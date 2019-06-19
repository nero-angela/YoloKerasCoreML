import coremltools
def getClasses(classes_path):
    with open('./data/light/classes.txt') as f:
        classes = f.read().splitlines()
    return classes

coreml_model = coremltools.converters.keras.convert('logs/000/model.h5',
                                                image_input_names='image',
                                                input_names="image",
                                                image_scale=1/255.0,
                                                class_labels=getClasses('data/light/classes.txt'),
                                                is_bgr=True)
coreml_model.save('logs/000/model.mlmodel')
