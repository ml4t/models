# API Reference

This reference is organized by stable import surface and by conceptual family.

## Recommended Imports

| Use case | Import surface |
|---|---|
| Notebook and exploratory work | `ml4t.models` |
| Structural model protocols | `ml4t.models.api` |
| Batch and result contracts | `ml4t.models.types` |
| Config objects | `ml4t.models.configs` |
| Cross-library handoff | `ml4t.models.integration` |

## Package Root

The package root re-exports the main model classes, configs, batches, results, and integration helpers.

::: ml4t.models
    options:
      show_root_heading: true

## Protocols

::: ml4t.models.api
    options:
      show_root_heading: true
      members:
        - LatentFactorModel
        - FactorForecaster
        - AssetMapper
        - AssetPredictionModel
        - StochasticDiscountFactorEstimator
        - PortfolioModel
        - PortfolioPostprocessor

## Typed Contracts

::: ml4t.models.types
    options:
      show_root_heading: true
      members:
        - PersistentPanelBatch
        - CrossSectionBatch
        - PortfolioSequenceBatch
        - FitSummary
        - LatentFactorState
        - FactorForecastResult
        - AssetForecastResult
        - AssetSignalResult
        - AssetWeightsResult
        - StochasticDiscountFactorState
        - PortfolioWeightsResult
        - LatentFactorPrediction
        - PortfolioPrediction

## Configs

::: ml4t.models.configs
    options:
      show_root_heading: true

## Pipelines

::: ml4t.models.pipelines
    options:
      show_root_heading: true
      members:
        - LatentFactorForecastPipeline
        - PipelineFitResult
        - PortfolioAllocationPipeline
        - PortfolioPipelineFitResult

## Integration

::: ml4t.models.integration
    options:
      show_root_heading: true

## Family Namespaces

| Namespace | Purpose |
|---|---|
| `ml4t.models.latent_factors` | structural latent-factor estimators |
| `ml4t.models.forecasters` | factor-premium forecasters |
| `ml4t.models.mappers` | asset-level mapping from factor forecasts |
| `ml4t.models.stochastic_discount_factor` | weight-native SDF estimation and return projections |
| `ml4t.models.asset_prediction` | direct asset-level predictors |
| `ml4t.models.portfolio` | end-to-end portfolio learners |

