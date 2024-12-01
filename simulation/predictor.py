import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

class Predictor:
    def __init__(self):
        self.model_dict = {}

    def get_predictor(self, prediction_length, context_length, num_feat_dynamic_real):
        key = (prediction_length, context_length, num_feat_dynamic_real)
        if key not in self.model_dict:
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
                prediction_length=prediction_length,
                context_length=context_length,
                patch_size="auto",
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=num_feat_dynamic_real,
                past_feat_dynamic_real_dim=0,
            )
            predictor = model.create_predictor(batch_size=1)
            self.model_dict[key] = predictor
        else:
            predictor = self.model_dict[key]
        return predictor

    def predict(self, target, feat_dynamic_real_extended, df_input_index, prediction_length):
        prediction_data = ListDataset(
            [{
                'start': df_input_index[0],
                'target': target,
                'feat_dynamic_real': feat_dynamic_real_extended
            }],
            freq='Y'
        )

        context_length = len(target)
        num_feat_dynamic_real = len(feat_dynamic_real_extended)
        predictor = self.get_predictor(prediction_length, context_length, num_feat_dynamic_real)
        forecasts = list(predictor.predict(prediction_data))
        return forecasts[0]
