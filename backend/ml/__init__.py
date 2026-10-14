"""FraudGuard machine-learning package (credit-card fraud detection).

The same modules are imported by the offline training pipeline and by the
FastAPI serving layer, so feature definitions, preprocessing and validation
cannot silently diverge between training and serving.
"""
