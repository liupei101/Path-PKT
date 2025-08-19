import sys

CLAM_PATH = "/path/to/CLAM"
sys.path.append(CLAM_PATH)

import numpy as np
import torch
import torch.nn.functional as F

from models.conch import tokenize, get_tokenizer


@torch.no_grad()
def zero_shot_classifier(model, classnames, templates, tokenizer=None, device='cuda'):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    zeroshot_weights = []
    for classnames_for_class in classnames:
        embeddings_for_class = []
        for classname in classnames_for_class:
            texts = [template.replace('CLASSNAME', classname) for template in templates]
            token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
            token_ids = token_ids.to(device)
            classname_embeddings = model.encode_text(token_ids)
            # classname_embeddings: [num_templates, embedding_dim]
            embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))

        # class_embedding: [num_classnames, num_templates, embedding_dim]
        class_embedding = torch.stack(embeddings_for_class, dim=0)
        # over all templates and classnames
        class_embedding = class_embedding.mean(dim=(0, 1))
        class_embedding /= class_embedding.norm()

        # class_embedding: [embedding_dim]
        zeroshot_weights.append(class_embedding)

    # zeroshot_weights: [embedding_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

@torch.no_grad()
def run_zeroshot(model, classifier, visual_feats):
    image_features = F.normalize(visual_feats, dim=-1)
    logits = image_features @ classifier
    probs = F.softmax(logits * model.logit_scale.exp().item(), dim=1)
    preds = logits.argmax(dim=1)

    return logits.cpu().numpy(), probs.cpu().numpy(), preds.cpu().numpy()
