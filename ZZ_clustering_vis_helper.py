from matplotlib import pyplot as plt


## Move this to clustering and remove plot_embeddings
def create_label_mapping(all_embeddings_np, all_labels_np, class_names):
    # Create label mappings
    if class_names and isinstance(class_names[0], tuple):
        label_mapping = {}
        superclass_mapping = {}
        for label in np.unique(all_labels_np):
            class_name, superclass = class_names[label]
            superclass_mapping[label] = superclass if superclass else class_name

            if label_type == 'super':
                label_mapping[label] = superclass
            elif label_type == 'class':
                label_mapping[label] = class_name
            elif label_type == 'both':
                label_mapping[label] = f"{class_name} ({superclass})"
    else:
        label_mapping = {label: str(label) for label in np.unique(all_labels_np)}

