# Design

Importance factors, seismic design category and the equivalent lateral force
procedure of ASCE 7-10 / 7-16 / 7-22, and the force-based design of
NEC-SE-DS 6.3. See the [design guide](../guides/design.md).

## ASCE 7

::: apeQuake.design.asce7
    options:
      members:
        - importance_factor
        - seismic_design_category
        - SDCResult
        - approximate_period
        - upper_limit_coefficient
        - seismic_response_coefficient
        - vertical_distribution_exponent
        - vertical_distribution
        - base_shear
        - ELFResult

## NEC-SE-DS

::: apeQuake.design.nec
    options:
      members:
        - importance_factor
        - approximate_period
        - limit_period_method_2
        - vertical_distribution_exponent
        - base_shear
        - NECELFResult
