from .rangerlars import RangerLars

optimizer = RangerLars(model.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
