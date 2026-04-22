def align_cells_between_modalities(df_peak, df_rna):
  
    # Retain only cells present in both modalities

    common_cells = df_peak.index.intersection(df_rna.index)

    df_peak_aligned = df_peak.loc[common_cells].copy()
    df_rna_aligned = df_rna.loc[common_cells].copy()

    return df_peak_aligned, df_rna_aligned
