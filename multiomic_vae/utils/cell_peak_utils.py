import pandas as pd



# load ATAC fragments file
# each row = one fragment mapped to a cell barcode
def load_fragments(path):
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["Chromosome", "Start", "End", "Barcode", "Count"]
    )



# load ATAC peaks file
# each row = one genomic peak (chr, start, end)
def load_peaks(path):
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["Chromosome", "Start", "End"]
    )



def filter_fragments(frags):
    # keep only standard chromosomes (no chrM, no contigs)
    valid_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    frags = frags[frags["Chromosome"].isin(valid_chroms)]
    print(f"After removing unwanted chromosomes: {len(frags):,}")

    # filter fragments by length (remove too short / too long)
    frag_len = frags["End"] - frags["Start"]
    frags = frags[(frag_len >= 50) & (frag_len <= 1000)]
    print(f"After fragment length filtering: {len(frags):,}")

    # remove low-quality cells (cells with few fragments)
    cell_counts = frags["Barcode"].value_counts()
    valid_cells = cell_counts[cell_counts >= 1000].index
    frags = frags[frags["Barcode"].isin(valid_cells)]
    print(f"After removing low-quality cells: {len(frags):,}")

    return frags.reset_index(drop=True)



def filter_peaks(peaks):
    # keep only standard chromosomes (no chrM, no contigs)
    valid_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    peaks = peaks[peaks["Chromosome"].isin(valid_chroms)].copy()

    print(f"Peaks kept after filtering: {len(peaks):,}")

    return peaks.reset_index(drop=True)



def assign_peak_ids(peaks):
    # make a meaningful peak id: chr_start_end
    peaks = peaks.copy()
    peaks["PeakID"] = (
        peaks["Chromosome"].astype(str)
        + "_"
        + peaks["Start"].astype(str)
        + "_"
        + peaks["End"].astype(str)
    )
    return peaks
  
  
  
def map_fragments_to_peaks(frags, peaks):
    # map fragments to peaks by genomic overlap
    # output = rows of (Barcode, PeakID) for every overlap

    mapped = []
    valid_chroms = sorted(frags["Chromosome"].unique())
    print(f"Processing {len(valid_chroms)} chromosomes...")

    for chrom in valid_chroms:
        # work chromosome by chromosome to keep things smaller
        frag_chr = frags[frags["Chromosome"] == chrom]
        peak_chr = peaks[peaks["Chromosome"] == chrom]

        if frag_chr.empty or peak_chr.empty:
            continue

        for _, p in peak_chr.iterrows():
            # overlap rule (NOT full containment)
            inside = frag_chr[
                (frag_chr["End"] > p["Start"]) &
                (frag_chr["Start"] < p["End"])
            ]

            if not inside.empty:
                tmp = inside[["Barcode"]].copy()
                tmp["PeakID"] = p["PeakID"]
                mapped.append(tmp)

        print(f"{chrom}: {len(mapped):,} chunks appended")

    merged = pd.concat(mapped, ignore_index=True) if mapped else pd.DataFrame(columns=["Barcode", "PeakID"])
    print(f"Final mapped rows: {len(merged):,}")

    return merged



def build_peak_matrix(mapped):
    # build cell × peak count matrix from mapped fragments
    # rows = cells (barcodes)
    # columns = peaks
    # values = number of overlapping fragments

    # count fragment overlaps per (cell, peak)
    counts = (
        mapped.groupby(["Barcode", "PeakID"])
              .size()
              .reset_index(name="Count")
    )

    # pivot to cell × peak matrix
    X_df = counts.pivot(
        index="Barcode",
        columns="PeakID",
        values="Count"
    )

    # fill missing combinations with zero counts
    X_df = X_df.fillna(0).astype("uint16")

    print("DataFrame built:", X_df.shape)

    return X_df
