from collections import defaultdict
from pathlib import Path
import sys
import os

import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gimmemotifs.motif import read_motifs
from gimmemotifs.comparison import MotifComparer
from gimmemotifs.cluster import cluster_motifs, MotifTree
from nopeak import NoPeakMotif
from scores import ScoreSet

from scipy.stats import spearmanr

def read_jaspar_motifs(input_path, tf):
    """
    Reads in JASPAR motif files in .jaspar format as gimmemotif Motif objects.

    Arguments
    ----------
    input_path : str
        Path to input files that contain JASPAR motifs
    tf : str
        Name of transcription factor the motifs are for

    Output
    ----------
    returns defaultdict with tf name as the key and value being a list of JASPAR motifs
    
    """
    motifs = defaultdict(list)

    for file in Path(f"{input_path}/").glob("*.jaspar"):
        motifs[str(tf)] += read_motifs(str(file), fmt="jaspar")

    return motifs

def read_nopeak_motifs(input_path, tf):
    """
    Reads in NoPeak motif files as gimmemotif Motif objects.

    Arguments
    ----------
    input_path : str
        Path to directory which contains the NoPeak motif files (sample_id.motifs.txt & sample_id.kmers.txt)
    tf : str
        Name of transcription factor the motifs are for

    Output
    ----------
    returns a defaultdict with tf name as the key and value being a list of NoPeak motifs

    """
    motifs = defaultdict(list)
    for file in Path(f"{input_path}/").glob("*_dedup.bam.motifs.txt"):
        motifs[str(tf)] += NoPeakMotif.from_file(
                str(file)
            )

    return motifs

def score_snps(data, motifs, tf, genome):
    # Return None if input data is empty, motifs is None or empty
    if (data.empty):
        return None

    if motifs is None:
        return None
     
    if (len(motifs[tf])==0):
        return None
    
    # Create score set object through gimmemotifs for all facors and motifs within SNP sites for genome hg19
    print("Creating score set object through gimmemotifs for all factors and motifs in ASB sites...")
    score_sets = { factor : ScoreSet(data, motif_list, genome) for factor, motif_list in motifs.items()}

    # Get best motif score with FPR < 0.05
    print("Pulling best motif score with FPR < 0.05 for all motifs in this set...")
    all_scores = { factor : set.get_best_scores(fpr=0.05) for factor, set in score_sets.items() }

    if all_scores[tf].empty:
        print(f"No sequences scored against motifs for {tf} found with FDR < 0.05")
        return None
    
    # Remove score_diff == 0
    scores = { factor : set[set.score_diff != 0] for factor, set in all_scores.items()}
    return scores

def pull_ic(motifs):
    """
    Get information content for a set of motifs.

    Arguments
    ----------
    motifs : defaultdict with the name of the tf as the key and values containing the list of gimmemotif Motif objects 
                (this should be the output of either read_jaspar_motifs or read_nopeak_motifs)

    Output
    ----------
    returns a pandas DataFrame with a row for each motif in the input and its information content.
    """
    df = pd.DataFrame(columns = ['motif','information_content'])
    for factor, motif_list in motifs.items():
        df = pd.DataFrame({'motif':[str(motif) for motif in motif_list], 
                           'information_content': [str(motif.information_content) for motif in motif_list]})

    return df

def pull_motif_metadata(motifs, kmer_count = False):
    """
    Pull metadata for motifs which includes information content and/or kmer count. 
    Kmer count is only available if your input motifs are from NoPeak.

    Arguments
    ----------
    motifs :  defaultdict 
        List of gimmemotif Motif objects for a set of motifs, with the key being set to the tf name.
        This should be the output of either read_jaspar_motifs or read_nopeak_motifs).
    kmer_count (default = False) : bool 
        Whether the output DataFrame should return kmer count as a column for each motif. 

    Output
    ----------
    returns a pandas DataFrame with a row for each motif in the input, its information content and potentially kmer count. 

    """
    df = pd.DataFrame(columns = ['motif','information_content'])
    for factor, motif_list in motifs.items():
        if kmer_count:
            df = pd.DataFrame({'motif':[str(motif) for motif in motif_list], 
                            'information_content': [str(motif.information_content) for motif in motif_list],
                            'kmer_count' : [str(motif.kmer_count) for motif in motif_list]})
        else:
            df = pd.DataFrame({'motif':[str(motif) for motif in motif_list], 
                            'information_content': [str(motif.information_content) for motif in motif_list]})

    return df

def get_snps(input_path, tf, filter_peak = True):
    """
    Read in ASB site CSV file output from the baal-nf pipeline.

    Arguments
    ----------
    input_path : str
        Path of the directory which contains the ASB sites. This will read them in based the tf name being present in the input files.
    tf : str
        Name of transcription factor these ASB sites were called for.

    Output
    ----------
    pandas DataFrame with ASB site information across all heterozygous SNPs and cell lines contained in your input directory for the tf-of-interest.

    """
    data = dd.read_csv(f"{input_path}/*{tf}*.csv", include_path_column=True)
    if filter_peak:
        data = data[data.peak]
    data["cell_line"] = data.apply(lambda row: Path(row.path).name.split("_")[0],  axis=1)
    data["tf"] = data.apply(lambda row: Path(row.path).name.split(".")[0].split("_")[1],  axis=1)
    data = data.compute().reset_index(drop=True)

    # To ensure there are no partial string matches
    data = data[data.tf == tf]

    return data

def filter_low_kmer_motifs(md, motifs, tf, kmer_threshold = 10, save_hist = True, save_logos = False, 
                            out_dir = ".", sub_dir = False):
    """
    Filter out NoPeak motifs with low KMER count.

    Arguments
    ----------
    md : pandas DataFrame
        DataFrame containing the KMER count information of the NoPeak motifs.
        This should be the output of pull_motif_metadata() with kmer_count = True.
    motifs : defaultdict
        List of gimmemotif Motif objects for a set of motifs, with the key being set to the tf name.
        This should be the output of either read_jaspar_motifs or read_nopeak_motifs). 
    tf : str
        Transcription factor name.
    kmer_threshold (default = 10): int
        KMER count threshold below which NoPeak motifs will be filtered out.
    save_hist (default = True) : bool
        Boolean value stating whether to save a histogram of all KMER counts for the input motifs.
    save_logos (default = False) : bool
        Boolean value stating whether to save logo plots for low KMER count motifs.
    out_dir (default = ".") : str
        Output directory to save plots.
    sub_dir (default = False) : bool
        Boolean value specifying whether you want to save the plots in a subdirectory based on the plot type.

    Output
    ----------
    returns
        1) a DataFrame which contians the metadata for motifs that have passed the KMER count threshold, and 
        2) a defaultdict for a List of Motif objects which have passed this threshold. 
    """
    # convert to type float for plotting
    md.kmer_count = md.kmer_count.astype(float)
    n_bins = int(round(md.shape[0]/20, 0) if round(md.shape[0]/20, 0) > 10 else 10)
    if save_hist:
        os.makedirs(f"{out_dir}", exist_ok = True)
        plt.figure(figsize=(20,10), dpi=1000)
        plt.hist(md.kmer_count, bins = n_bins)
        plt.savefig(f"{out_dir}/histogram_kmer_counts.png")

    # plot logo of the lowest kmer_count motifs
    if save_logos:
        low_kmer_motif_names = md[md.kmer_count < kmer_threshold].motif.values
        low_kmer_motifs = [motif for motif in motifs[tf] if str(motif) in low_kmer_motif_names]

        if sub_dir:
            out_dir = f"{out_dir}/low_kmer_logos"
        os.makedirs(f"{out_dir}", exist_ok = True)
        for i, motif in enumerate(low_kmer_motifs):
            plt.figure(figsize=(5,5), dpi=100)
            motif.trim().plot_logo()
            plt.savefig(f"{out_dir}/{tf}_low_kmer_NoPeak_motif_logo_{i}.png")
        
    high_kmer_motif_md = md[md.kmer_count >= kmer_threshold]
    high_kmer_motifs = { tf : [motif for motif in motifs[tf] if str(motif) in high_kmer_motif_md.motif.values]}

    return high_kmer_motif_md, high_kmer_motifs

def cluster_high_kmer_motifs(motifs, tf, ncpus = 1, out_dir = ".", save_logos = False, sub_dir = False):
    """
    For all remaining high KMER motifs, cluster this to remove redundant motifs.
    This function first clusters all input motifs, and then extracts the clustered motifs.
    Optional to save the logos of all clustered motifs.

    Arguments
    ----------
    motifs : defaultdict
        List of gimmemotif Motif objects for a set of motifs, with the key being set to the tf name.
        This should be the second output of filter_low_kmer_motifs().
    tf :  str
        Transcription factor name.
    ncpus (default = 1) : int
        Number of CPUs to use for clustering motifs.
    out_dir (default = ".") : str
        Output directory to save logos in (if specified).
    save_logos (default = False) : bool
        Boolean value specifying whether to save the motif logo plots for the clustered motifs.
    sub_dir (default = False) : bool
        Boolean value specifying whether you want to save the plots in a subdirectory based on the plot type.

    Output
    ----------
    defaultdict with a List of clustered motifs, with the key being set to the tf name.
    """
    # Check if there is only 1 motif, or none
    if (len(motifs[tf]) == 0):
        print(f"No NoPeak motifs found with KMER count > 10 for {tf}. Exiting...")
        exit(0)
    elif (len(motifs[tf]) == 1):
        return motifs
    
    # Cluster using gimmemotifs clustering method and MotifTree object
    print("Clustering high KMER motifs...")
    clustered_motifs = cluster_motifs(
        motifs[tf],
        trim_edges=True,
        ncpus=ncpus,
        metric="seqcor",
        threshold=0.7,
        pval=False,
    )

    motif_cluster = clustered_motifs.get_clustered_motifs()

    # Save clustered motif logos
    if save_logos:
        if sub_dir:
            out_dir = f"{out_dir}/clustered_logos"
        os.makedirs(f"{out_dir}", exist_ok = True)
        for i, motif in enumerate(motif_cluster):
            motif_name = str(motif)
            plt.figure(figsize=(5,5), dpi=100)
            motif.trim().plot_logo()
            plt.savefig(f"{out_dir}/{tf}_clustered_motif_{motif_name}.png") 
    motif_dict = { tf : motif_cluster}

    return motif_dict

def find_high_quality_motifs(scores, corrs, motifs, tf, return_dict = True):
    """
    Filter correlations DataFrame to only include high-quality motifs.
    These are defined as those with:
        1) a significant Spearman's rho correlation coefficient between the corrected allelic ratio and the chaneg in motif score.
        2) a positive Spearman's rho correlation coefficient.
        3) a non-NA p-value.
    
    Arguments
    ----------
    scores : scores Dict containing scores for TFs-of-interest
        All scores to a set of motifs for every SNP-of-interest.
    corrs : pandas DataFrame
        Spearman's correlation coefficient and associated p-value for every motif-of-interest. 
    motifs : defaultdict
        List of gimmemotif Motif objects for a set of motifs, with the key being set to the tf name.
        This should be the output of either read_jaspar_motifs() or cluster_high_kmer_motifs(). 
    tf : str
        Transcription factor name.

    Output 
    ----------
    returns:
        1) a list of Motif objects for the motifs which fit the criteria of "high-quality".
        2) a pandas DataFrame containing the scores for all SNPs to all motifs-of-interest, filtered for high-quality motifs.
    """
    if scores is None:
        return None, None
    
    df = scores[tf]
    filt = []
    significant_motifs = corrs[
        (corrs.pval <= 0.05) & 
        (corrs.spearmans_rho > 0) & 
        (corrs.spearmans_rho != 1.0) # low SNP count motifs are reporting 1.0 even when this is not true
        ].motif.values
    table_filt = df[df['motif'].isin(significant_motifs)] 
    for motif in motifs[tf]:
        if str(motif) in significant_motifs:
            filt.append(motif)

    if table_filt.shape[0]==0:
        sys.stderr.write(f"No significant motifs found for {tf}.\n")

    if not return_dict:
        return filt, table_filt
    filt_dict = { tf : filt }
    return filt_dict, table_filt


def find_motif_matches(motifs, tf, save_accessory = True, out_dir = "."):
    """
    Match a set of NoPeak motifs to known JASPAR motifs and classify them into three groups:
        1) Matches expected JASPAR motif (redundant)
        2) Matches JASPAR motif that is not expected (accessory)
        3) Matches no known JASPAR motifs (de novo)

    Arguments
    ----------
    motifs : defaultdict
        List of gimmemotif Motif objects for a set of motifs, with the key being set to the tf name.
        This should be the output of find_high_quality_motifs().  
    tf : str
        Transcription factor name.
    save_accessory (default = True) : bool
        Boolean value which specifies whether to save a dataframe of the JASPAR matches for the accessory motif category.
    out_dir (default = ".") : str
        Output directory to save this table.

    Output
    ----------
    returns:
        1) List of motif names which are redundant
        2) List of motif names which are accessory
        3) List of motif names which are de novo 
    """
    # Access motif list for tf-of-interest
    motif_list = motifs[tf]
    if len(motif_list)==0:
        return None, None, None

    mc = MotifComparer()
    compare_motifs = mc.get_all_scores(motifs=motif_list, dbmotifs=read_motifs("JASPAR2022_vertebrates"), match="partial", metric="seqcor", combine="mean")

    a=[]
    for i in compare_motifs.keys():
        b = pd.DataFrame.from_dict(compare_motifs[i]).T
        b["query"] = i
        a.append(b)

    summary = pd.concat(a,axis=0)
    summary.columns = ['score','position','strand','NoPeak_motif']
    summary = summary.assign(TF = [tf for idx1, idx2, idx3, tf in summary.index.str.split('.')])
    summary_human = summary[summary['TF'].str.isupper()]

    # Now identify when you get a match to the expected TF
    matches_expected = np.unique(
        summary_human[
            (summary_human.TF==tf) & 
            (summary_human.score >= 0.7)
            ].NoPeak_motif.values
    )

    matches_new = np.unique(
        summary_human[
            (summary_human.score >= 0.7) & 
            np.logical_not(summary_human.NoPeak_motif.isin(matches_expected))
            ].NoPeak_motif.values
    )

    # Find NoPeak motifs that match no known JASPAR motifs
    motif_names = ['_'.join(str(x).split('_')[:-1]) for x in motif_list] # remove trailing characters from motif name
    matches_none = np.unique([motif for motif in motif_names if motif not in 
                    np.concatenate((matches_expected,matches_new), axis = 0)])

    if save_accessory:
        motif_matches = summary_human[
                            (summary_human.score >= 0.7) & 
                            np.logical_not(summary_human.NoPeak_motif.isin(matches_expected))
                            ] 
        if motif_matches.shape[0] > 0:
            motif_matches.to_csv(f"{out_dir}/{tf}_accessory_motif_JASPAR_matches.csv")

    return matches_expected, matches_new, matches_none

def save_logo_plot(motifs, motif_list, tf, motif_group, out_dir = ".", sub_dir = False, save = False):
    """
    Save NoPeak logo plots.

    Arguments
    ----------
    motifs : List
        List of motif names for which you would like to plot their LOGOs.
    motif_list : List
        List of Motif objects which contain motif you would like to plot the LOGOs for.
    tf : str
        Transcription factor name.
    motif_group : str
        Either redundant, accessory or de_novo. This is used for naming output files/directories.
    out_dir (default = ".") : str
        Output directory to save LOGO plots.
    sub_dir (default = False) : bool
        Specifies whether to save these plots in subdirectories dependent on motif category.
    save (default = False) : bool
        Specifies whether to save these plots or not.

    Output
    ----------
    Does not return anything, but saves the plots to the specified output directory. 
    """
    if motifs is None:
        return
    if save:
        if len(motifs) != 0:
            motifs_to_plot = []
            for motif_name in motifs:
                for motif in motif_list:
                    if str(motif).startswith(motif_name):
                        motifs_to_plot.append(motif)

            if sub_dir:
                out_dir = f"{out_dir}/{motif_group}"
            os.makedirs(f"{out_dir}", exist_ok = True)
            for i, motif in enumerate(motifs_to_plot):
                plt.figure(figsize=(5,5), dpi=100)
                motif.trim().plot_logo()
                plt.savefig(f"{out_dir}/{tf}_{motif_group}_NoPeak_motif_logo_{i}.png")

def add_nopeak_information(df, corrs, redundant_motif_list, accessory_motif_list, denovo_motif_list):
    """
    Add NoPeak motif categorization into SNP-level and motif-level DataFrames.

    Arguments
    ----------
    df : pandas DataFrame
        Contains the scores for all SNPs to all motifs-of-interest, filtered for high-quality motifs. 
    corrs : pandas DataFrame
        Contains the correlation coefficients for all motifs-of-interest.
    redundant_motif_list : List
        List of motif names that are redundant with the canonical JASPAR motif.
    accessory_motif_list : List
        List of motif names that are defined as accessory.
    denovo_motif_list : List
        List of motif names that are de novo.

    Output
    ----------
    pandas DataFrame which ontains the scores for all SNPs to all motifs-of-interest with new column added called NoPeak_mapping.
    """
    nopeak_map = []
    sig_motifs = corrs[(corrs.pval <= 0.05) & (corrs.spearmans_rho > 0)]
    for i, motif in enumerate(df.motif.values):
        # remove trailing motif characters for motif_list query
        query = '_'.join(motif.split('_')[0:2]) if motif.startswith('Average') else '_'.join(motif.split('_')[0:3]) 
        
        # If motif not significant, label as NA
        if motif not in sig_motifs.motif.values.tolist():
            nopeak_map.append(np.nan)
        elif query in redundant_motif_list:
            nopeak_map.append(f"NoPeak redundant motif")
        elif query in accessory_motif_list:
            nopeak_map.append("NoPeak accessory motif")
        elif query in denovo_motif_list:
            nopeak_map.append("NoPeak de novo motif")
        else:
            raise ValueError("Motif does not fit any known category.")
    df["NoPeak_mapping"] = nopeak_map 

    return df

def compute_correlations(scores_dict):
    """
    Compute Spearman's correlation coefficient between direction of binding (Corrected.AR) and change in motif score (score_diff).

    Arguments
    ----------
    scores_dict : defaultdict
         defaultdict with the name of the tf as the key and values containing the list of gimmemotif Motif objects 
         This should be the output of either read_jaspar_motifs() or read_nopeak_motifs().

    Output
    ----------
    pandas DataFrame containing each motif in your input, with the Spearman's rho correlation coefficient and associated p-value. 
    """
    if scores_dict is None:
        corrs = pd.DataFrame(columns = ['factor', 'motif', 'spearmans_rho', 'pval'])
        return corrs

    corr_dict = defaultdict(dict)
    for factor, score_set in scores_dict.items():
        for motif, group in score_set.groupby("motif"):
            correlation = spearmanr(group["Corrected.AR"],group.score_diff)
            corr_dict[factor][motif] = [correlation[0],correlation[1]]
            print(f"{factor}, {motif}: {correlation}") 

    # Export dataframe with motif/factor/correlation coefficient mapping
    corrs = pd.DataFrame(columns = ['factor','motif','spearmans_rho','pval'])

    for factor, vals in corr_dict.items():
        roes = [t[0] for t in vals.values()]
        pvals = [p[1] for p in vals.values()]
        tmp = pd.DataFrame({'factor': factor, 'motif' : vals.keys(), 'spearmans_rho' : roes, 'pval' : pvals})
        corrs = corrs.append(tmp)

    # Drop any missing values
    corrs = corrs.dropna(subset = ['pval'])
    corrs = corrs[corrs.spearmans_rho != 1.0] # This only happens when very few SNPs map and isn't accurate

    return corrs

def compute_snp_counts(scores_dict, asb, motif_tf, asb_tf):
    """
    Compute the number of SNPs that map to a given motif.

    Arguments
    ----------
    df : default Dict
        Scores dict all SNPs mapped to all high-quality motifs and their associated scores.
    asb : pandas DataFrame
        DataFrame containing all ASB sites.
        This should be the output of get_asb().
    motif_tf : str
        Transcription factor name.
    asb_tf : str
        Transcription factor name. 
    [These TF names are variable in case you would like to check ASBs from one TF to motifs of a different TF]

    Output
    ----------
    pandas DataFrame which summarizes the SNP counts that map to all motifs-of-interest. 
    """
    if scores_dict is None:
        return pd.DataFrame(columns = ['motif', 'counts', 'fraction_of_snps', 'tf_motif', 'tf_asb'])
    df = scores_dict[motif_tf]
    rsid_counts = {factor : len(set.ID.unique()) for factor, set in asb.groupby("tf")}
    snp_cts = []
    id_counts = pd.DataFrame(df.pivot(columns=("motif"), values=("ID")).apply(lambda series: len(series.dropna().unique()), axis=0), columns=["counts"]).sort_values(by="counts", ascending=False).reset_index()
    id_counts["fraction_of_snps"] = id_counts["counts"]/rsid_counts[asb_tf]
    id_counts["tf_motif"] = motif_tf
    id_counts["tf_asb"] = asb_tf
    snp_cts.append(id_counts)
    snp_cts = pd.concat(snp_cts)
    return snp_cts

def compile_scores_data(scores_dict, tf):
    if scores_dict is None:
        return pd.DataFrame(columns = ['ID', 'CHROM', 'POS', 'REF', 'ALT', 'REF.counts', 'ALT.counts',
       'Total.counts', 'AR', 'RMbias', 'RAF', 'Bayes_lower', 'Bayes_upper',
       'Bayes_SD', 'conf_0.99_lower', 'conf_0.99_upper', 'Corrected.AR',
       'isASB', 'peak', 'path', 'cell_line', 'tf', 'ref_seq', 'alt_seq',
       'motif', 'motif_pos', 'motif_strand', 'ref_score', 'alt_score',
       'score_diff','High_quality_motif'])
    df = scores_dict[tf].copy()
    df["High_quality_motif"] = df["motif"].isin(df["motif"].unique())
    return df

def compile_motif_data(corr_df, ic_df, snp_counts_df, tf):
    motif_df = corr_df.set_index('motif').join(ic_df.set_index('motif'))
    motif_df = motif_df.drop(['factor'],axis=1)
    motif_df["high_quality"] = (motif_df.pval <= 0.05) & (motif_df.spearmans_rho > 0)
    motif_df = pd.concat([snp_counts_df.set_index('motif'),motif_df], axis = 1)
    motif_df = motif_df[["counts","fraction_of_snps","spearmans_rho","pval","information_content","high_quality"]]
    motif_df["tf"] = tf
    return motif_df

def filter_for_high_quality_motifs(df):
    """
    Filter DataFrame for motifs that are high-quality

    Arguments
    ----------
    df : pandas DataFrame
        DataFrame contianing all SNPs mapped to all motifs and their associated scores.
     
    Output
    ----------
    pandas DataFrame that has been filtered for only "high-quality" motifs. 
    """
    return df[
        df.High_quality_motif
    ].copy()

def filter_asbs(df, concordant = True):
    """
    Filter dataframe for ASB sites that are either:
        1) Concordant or
        2) Discordant

    Arguments
    ---------- 
    df : pandas DataFrame
        pandas DataFrame that has been filtered for only "high-quality" motifs. 
    concordant (default = True) : bool
        Specifies whether you would like to filter for concordant or discordant ASB sites.
        * Concordant is defined as - the direction of binding is concordant with the change in motif score.
        * Discordant ASB sites display the opposite behaviour. 

    Output
    ---------- 
    pandas DataFrame where rows that do not meet the criteria specified are filtered out.
    """
    if concordant:
        sub_df = df[
            df.isASB
            & (
                ((df["Corrected.AR"] > 0.5) & (df.score_diff > 0))
                | ((df["Corrected.AR"] < 0.5) & (df.score_diff < 0))
            )
        ].copy()
    else:
        sub_df = df[
            df.isASB
            & (
                ((df["Corrected.AR"] < 0.5) & (df.score_diff > 0))
                | ((df["Corrected.AR"] > 0.5) & (df.score_diff < 0))
            )
        ].copy()
    sub_df["Concordant"] = concordant
    return sub_df

def get_snp_data(input_path, tf):
    data = dd.read_csv(f"{input_path}/*{tf}.withPeaks.csv", include_path_column=True)
    data["cell_line"] = data.apply(lambda row: Path(row.path).name.split("_")[0],  axis=1)
    data["tf"] = data.apply(lambda row: Path(row.path).name.split(".")[0].split("_")[1],  axis=1)
    data = data.drop(['path'], axis = 1)
    peak_data = data[data.peak].copy()
    data = data.compute().reset_index(drop=True)
    peak_data = peak_data.compute().reset_index(drop=True)

    return peak_data, data

def get_nopeak(input_path, tf, motif_mode):
    if ((motif_mode == "noNoPeak") | (motif_mode == "noMotifs")):
        colnames = ['ID', 'CHROM', 'POS', 'REF', 'ALT', 'REF.counts', 'ALT.counts', 'Total.counts', 
        'AR', 'RMbias', 'RAF', 'Bayes_lower', 'Bayes_upper', 'Bayes_SD', 'conf_0.99_lower', 'conf_0.99_upper', 
        'Corrected.AR', 'isASB', 'peak', 'path', 'cell_line', 'tf', 'ref_seq', 'alt_seq', 'motif', 'motif_pos', 
        'motif_strand', 'ref_score', 'alt_score', 'score_diff', 'tf_motif', 'High_quality_motif', 'NoPeak_mapping', 'Concordant']
        data = pd.DataFrame(columns = colnames)
    elif ((motif_mode == "allMotifs") | (motif_mode == "noJaspar")):
        data = pd.read_csv(f"{input_path}/{tf}_ASBs_NoPeak_Motifs_AR_score_diff.csv")
    else:
        sys.stderr("No valid mode given, Please specify either allMotifs, noMotifs, noNoPeak or noJaspar.")
        exit(1)
    
    data = data.convert_dtypes()
    data = data[data.tf == tf]
    return data

def get_jaspar(input_path, tf, motif_mode):
    if ((motif_mode == "noJaspar") | (motif_mode == "noMotifs")):
        colnames = ['ID', 'CHROM', 'POS', 'REF', 'ALT', 'REF.counts', 'ALT.counts', 'Total.counts', 
        'AR', 'RMbias', 'RAF', 'Bayes_lower', 'Bayes_upper', 'Bayes_SD', 'conf_0.99_lower', 'conf_0.99_upper', 
        'Corrected.AR', 'isASB', 'peak', 'path', 'cell_line', 'tf', 'ref_seq', 'alt_seq', 'motif', 'motif_pos', 
        'motif_strand', 'ref_score', 'alt_score', 'score_diff', 'tf_motif', 'High_quality_motif', 'NoPeak_mapping', 'Concordant']
        data = pd.DataFrame(columns = colnames)
    elif ((motif_mode == "allMotifs") | (motif_mode == "noNoPeak")):
        data = pd.read_csv(f"{input_path}/{tf}_ASBs_JASPAR_Motifs_AR_score_diff.csv")
    else:
        sys.stderr("No valid mode given, Please specify either allMotifs, noMotifs, noNoPeak or noJaspar.")
        exit(1)
    
    data = data.convert_dtypes()
    data = data[data.tf == tf]
    return data

def subset_asbs(jaspar, nopeak, tf, concordant = True, isasb = True):
    sub_jaspar = jaspar[
        (jaspar.High_quality_motif) & 
        (jaspar.isASB == isasb) &
        (jaspar.tf == tf)
        ].copy()
    sub_nopeak = nopeak[
        (nopeak.High_quality_motif) & 
        (nopeak.isASB == isasb) &
        (nopeak.tf == tf) &
        (nopeak.NoPeak_mapping != 'NoPeak redundant motif')
        ].copy()

    if isasb:
        sub_jaspar = sub_jaspar[sub_jaspar.Concordant == concordant]
        sub_nopeak = sub_nopeak[sub_nopeak.Concordant == concordant]

    return sub_jaspar, sub_nopeak 

def classify_jaspar_concordant(snp, cell_line, concordant_jaspar, snp_data):
    row = concordant_jaspar.index[
        (concordant_jaspar['ID'] == snp) &
        (concordant_jaspar['cell_line'] == cell_line)
    ]
    sub = concordant_jaspar.loc[row].copy()
    motifs = ";".join(np.unique(sub.motif.values))
    out = sub[snp_data.columns.tolist() + ['High_quality_motif', 'Concordant']].drop_duplicates()
    out['Motifs'] = motifs
    out['Motif_group'] = 'JASPAR'
    if out.shape[0] > 1:
        raise ValueError("More than one transcription factor present in this table")
    return out

def classify_nopeak_concordant(snp, cell_line, concordant_nopeak, snp_data):
    row = concordant_nopeak.index[
        (concordant_nopeak['ID'] == snp) &
        (concordant_nopeak['cell_line'] == cell_line) &
        (concordant_nopeak['NoPeak_mapping'] == "NoPeak accessory motif")
    ]
    if len(row) > 0:
        group = "NoPeak_accessory_motif"
    elif len(row) == 0:
        row = concordant_nopeak.index[
            (concordant_nopeak['ID'] == snp) &
            (concordant_nopeak['cell_line'] == cell_line) &
            (concordant_nopeak['NoPeak_mapping'] == "NoPeak de novo motif")
        ]
        group = "NoPeak_denovo"
    sub = concordant_nopeak.loc[row].copy()
    motifs = ";".join(np.unique(sub.motif.values))
    out = sub[snp_data.columns.tolist() + ['High_quality_motif', 'Concordant']].drop_duplicates()
    out['Motifs'] = motifs
    out['Motif_group'] = group
    if out.shape[0] > 1:
        raise ValueError("More than one transcription factor present in this table")
    return out

def classify_discordant(snp, cell_line, discordant_jaspar, discordant_nopeak, snp_data):
    row_jasp = discordant_jaspar.index[
        (discordant_jaspar['ID'] == snp) &
        (discordant_jaspar['cell_line'] == cell_line)
    ]
    row_nopeak = discordant_nopeak.index[
        (discordant_nopeak['ID'] == snp) &
        (discordant_nopeak['cell_line'] == cell_line)
    ]
    subjasp = discordant_jaspar.loc[row_jasp].copy()
    subnopeak = discordant_nopeak.loc[row_nopeak].copy()
    if subjasp.shape[0] != 0 and subnopeak.shape[0] != 0:
        group = "JASPAR and NoPeak"
    elif subjasp.shape[0] != 0:
        group = "JASPAR"
    else:
        group = "NoPeak"
    sub = pd.concat([subjasp, subnopeak])
    motifs = ";".join(np.unique(sub.motif.values))
    out = sub[snp_data.columns.tolist() + ['High_quality_motif', 'Concordant']].drop_duplicates()
    out['Motifs'] = motifs
    out['Motif_group'] = group
    if out.shape[0] > 1:
        raise ValueError("More than one transcription factor present in this table")
    return out

def classify_non_asb(snp, cell_line, snps_jaspar, snps_nopeak, snp_data):
    row_jasp = snps_jaspar.index[
        (snps_jaspar['ID'] == snp) &
        (snps_jaspar['cell_line'] == cell_line)
    ]
    row_nopeak = snps_nopeak.index[
        (snps_nopeak['ID'] == snp) &
        (snps_nopeak['cell_line'] == cell_line)
    ]
    subjasp = snps_jaspar.loc[row_jasp].copy()
    subnopeak = snps_nopeak.loc[row_nopeak].copy()
    if subjasp.shape[0] != 0 and subnopeak.shape[0] != 0:
        group = "JASPAR and NoPeak"
    elif subjasp.shape[0] != 0:
        group = "JASPAR"
    else:
        group = "NoPeak"
    sub = pd.concat([subjasp, subnopeak])
    motifs = ";".join(np.unique(sub.motif.values))
    out = sub[snp_data.columns.tolist() + ['High_quality_motif', 'Concordant']].drop_duplicates()
    out['Motifs'] = motifs
    out['Motif_group'] = group
    return out

def classify_no_motif(index, snp_data):
    out = snp_data.loc[[index]].copy()
    out['High_quality_motif'] = pd.NA
    out['Concordant'] = pd.NA
    out['Motifs'] = pd.NA
    out['Motif_group'] = pd.NA
    return out

def map_snp_by_row(row, concordant_jaspar, concordant_nopeak, discordant_jaspar, discordant_nopeak, snps_jaspar, snps_nopeak, snp_data):
    index = row.name
    snp = row['ID']
    cell_line = row['cell_line']

    if snp in concordant_jaspar[concordant_jaspar.cell_line == cell_line].ID.values:
        return classify_jaspar_concordant(snp, cell_line, concordant_jaspar, snp_data)
    elif snp in concordant_nopeak[concordant_nopeak.cell_line == cell_line].ID.values:
        return classify_nopeak_concordant(snp, cell_line, concordant_nopeak, snp_data)
    elif snp in np.append(discordant_jaspar[discordant_jaspar.cell_line == cell_line].ID.values,
                          discordant_nopeak[discordant_nopeak.cell_line == cell_line].ID.values):
        return classify_discordant(snp, cell_line, discordant_jaspar, discordant_nopeak, snp_data)
    elif snp in np.append(snps_jaspar[snps_jaspar.cell_line == cell_line].ID.values,
                          snps_nopeak[snps_nopeak.cell_line == cell_line].ID.values):
        return classify_non_asb(snp, cell_line, snps_jaspar, snps_nopeak, snp_data)
    else:
        return classify_no_motif(index, snp_data)

def classify_asb_quality(row):
    if pd.notna(row['Concordant']) and row['Concordant']:
        if row['peak']:
            return 'High'
        else:
            return 'Medium'
    else:
        return pd.NA
    