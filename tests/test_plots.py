"""Plot axes must preserve recorded LS and comparable score units."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plots
import diagnostics
import plot
from config import BASE_DCS, HIST_DCS, DM_DCS, MODEL_MES, REGIONS


def frame():
    result = pd.DataFrame({
        "dataset": ["sample"] * 5,
        "run_number": [391550, 391550, 393240, 398902, 399000],
        "ls_number": [1, 4, 1, 1, 1],
        "label": ["good", "good", "bad", "good", "good"],
        "split": ["test"] * 5,
        "dcs_pass": [True, True, True, True, False],
        "anomaly_score": [2., np.nan, 4., 8., 10.],
        "flagged": [False, pd.NA, True, True, True],
    })
    for i in range(8):
        result[f"N_{i}"] = [5, 0, 20, 5, 5]
    return result


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_run_trend_preserves_the_requested_cms_run_label(self):
        label = 'Run 393240 (Good), 2025C'
        scores = {'Feature AE': np.ones(5), 'Histogram AE': np.full(5, 2.)}
        counts = {'Global': np.arange(1, 6), 'Standalone': np.arange(6, 11)}
        with patch.object(plots, 'save') as save:
            plots.plot_run_trend(np.arange(1, 6), counts, np.full(5, 20.),
                                 scores, 'unused.png', label=label)
        fig, _ = save.call_args.args
        self.assertIn(label, [text.get_text() for text in fig.axes[0].texts])
        self.assertEqual(fig.axes[0].get_legend_handles_labels()[1], list(counts))
        self.assertEqual(fig.axes[1].get_legend_handles_labels()[1], list(scores))

    def test_comparison_and_lowstats_derive_the_me_count_from_saved_columns(self):
        source = frame().drop(columns=[f"N_{i}" for i in range(3, 8)])
        aligned = plots._aligned_test({"model": source})["model"]
        self.assertEqual(list(aligned.filter(regex=r"^N_\d+$")), ["N_0", "N_1", "N_2"])
        self.assertEqual(plots._lowstat_rates(aligned, 2.).loc["1–10", "recorded"], 1)

    def test_diagnostics_find_physical_mes_after_reordering(self):
        names = list(reversed(MODEL_MES))
        run = REGIONS['B'][0]
        metadata = pd.DataFrame(dict(dataset=['sample'] * 2, run_number=[run] * 2,
                                     ls_number=[2, 1], label=['good'] * 2,
                                     split=['train'] * 2, dcs_pass=[True] * 2, era=['2025C'] * 2))
        dataset = dict(frame=metadata, me_names=names,
                       histograms=[np.array([[1, 3], [3, 1]]) for _ in names],
                       axes=[{'edges': np.array([0., 10., 20.])} for _ in names],
                       root_entries=np.tile(np.arange(1, len(names)+1), (2, 1)),
                       totals=np.full((2, len(names)), 4))
        frames = {name: metadata.assign(anomaly_score=[2., 4.]) for name in (BASE_DCS, HIST_DCS, DM_DCS)}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(diagnostics, 'OUT', Path(temporary)), patch.object(diagnostics, 'plot_run_trend') as draw:
                diagnostics.write_diagnostics(dataset, frames, {name: 2. for name in frames})
            ls, counts, mean_pt, scores, _ = draw.call_args.args
            np.testing.assert_array_equal(ls, [1, 2])
            phi = names.index('Muons__MuonRecoAnalyzer__GlbMuon_Glb_phi')
            np.testing.assert_array_equal(counts['GlbMuon ROOT fills'], [phi + 1] * 2)
            np.testing.assert_allclose(mean_pt, [7.5, 12.5])
            np.testing.assert_allclose(scores['DM-K1'], [2., 1.])
            self.assertEqual(draw.call_args.kwargs['label'], f'Run {run} (Good), 2025C')
            saved = pd.read_csv(Path(temporary)/'predict/diagnostics/region_b.csv')
            self.assertEqual(saved.ls_number.tolist(), [1, 2])

    def test_default_raw_plot_recipe_groups_cached_mes_by_example(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/'runs.csv'
            pd.DataFrame(dict(dataset=['/Muon0/Run2025C-PromptReco-v1/DQMIO'] * 3,
                              run_number=[1, 2, 3])).to_csv(manifest, index=False)
            for run in (1, 2, 3):
                folder = root/str(run)
                folder.mkdir()
                for me in ('Muons__first', 'Muons__second'):
                    (folder/f'{me}.parquet').touch()
            with patch.object(plot, 'EXAMPLES', {'good': (1,), 'bad': (2, 3)}), patch.object(plot, 'REGIONS', {'A': (2, 3)}), \
                    patch.object(plot, 'MANIFEST', manifest), patch.object(plot, 'raw_folder', side_effect=lambda _, run: root/str(run)), \
                    patch.object(plot, 'plot_me') as draw_me, patch.object(plot, 'plot_entries') as draw_entries:
                plot.plot_examples()
            self.assertEqual([len(call.args[0]) for call in draw_me.call_args_list], [1, 1, 1, 1, 1, 1, 2, 2])
            self.assertEqual([call.args[1] for call in draw_entries.call_args_list], [None, None, None, 'A'])

    def test_unknown_certification_uses_a_separate_background_and_legend(self):
        source = frame()
        source.loc[0, 'label'] = 'unknown'
        with patch.object(plots, 'save') as save:
            plots.score_view(source, 2., 'unused.png')
        fig, _ = save.call_args.args
        ax = fig.axes[0]
        from matplotlib.colors import to_rgba
        np.testing.assert_allclose(ax.patches[0].get_facecolor(), to_rgba(plots.GREY))
        self.assertIn('Unknown', [text.get_text() for text in ax.get_legend().texts])

    def test_recorded_ls_axis_does_not_close_unscored_gap_and_normalizes_threshold(self):
        with patch.object(plots, "save") as save:
            plots.score_view(frame().iloc[::-1], 2., "unused.png", regions=True)
        fig, _ = save.call_args.args
        ax = fig.axes[0]
        points = ax.collections[0].get_offsets()
        np.testing.assert_allclose(points[:, 0], [0, 2, 3])
        np.testing.assert_allclose(points[:, 1], [1, 2, 4])
        self.assertEqual(ax.get_xlim(), (-.5, 3.5))
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 1])
        self.assertTrue(any("1 unscored LS" in text.get_text() for text in ax.texts))
        self.assertTrue({"A", "B", "D"}.issubset({text.get_text() for text in ax.texts}))

    def test_comparison_alignment_keeps_missing_rows_on_common_ls_axis(self):
        first = frame()
        second = first.drop(index=0).copy()
        second["anomaly_score"] *= 3
        aligned = plots._aligned_test({"first": first, "second": second})
        pd.testing.assert_frame_equal(aligned["first"][plots.IDS], aligned["second"][plots.IDS])
        self.assertEqual(len(aligned["first"]), 4)
        self.assertTrue(np.isnan(aligned["second"].anomaly_score.iloc[0]))
        np.testing.assert_allclose(plots._values(aligned["first"], 2)[2:],
                                   plots._values(aligned["second"], 6)[2:])

    def test_display_order_is_run_then_ls_even_when_dataset_names_sort_differently(self):
        source = frame()
        source.loc[source.run_number.eq(391550), "dataset"] = "z_first_run"
        source.loc[source.run_number.eq(398902), "dataset"] = "a_last_run"
        aligned = plots._aligned_test({"model": source.iloc[::-1]})["model"]
        self.assertEqual(aligned.run_number.tolist(), [391550, 391550, 393240, 398902])
        self.assertEqual(aligned.ls_number.tolist(), [1, 4, 1, 1])

    def test_lowstats_excludes_special_controls_and_retains_empty_unknown_ls(self):
        source = frame()
        source.loc[0, "N_7"] = 0
        rates = plots._lowstat_rates(source, 2.)
        self.assertEqual(rates.loc["1–10", "recorded"], 1)
        self.assertEqual(rates.loc["1–10", "fraction"], 0)
        self.assertEqual(rates.loc["0", "recorded"], 1)
        self.assertEqual(rates.loc["0", "scored"], 0)
        self.assertTrue(np.isnan(rates.loc["0", "fraction"]))

    def test_comparison_shares_log_limits_and_omits_empty_categories_and_repeated_counts(self):
        frames = {}
        for i in range(6):
            source = frame()
            source["anomaly_score"] *= 10. ** i
            source.loc[1, [f"N_{j}" for j in range(8)]] = 20
            frames[f"model_{i}"] = source
        with patch.object(plots, "save") as save:
            plots.plot_comparison(frames, {name: 2. for name in frames}, "unused")
        figures = {Path(path).name: fig for fig, path in (call.args for call in save.call_args_list)}
        axes = figures["test_log.png"].axes
        self.assertEqual(len(axes), 6)
        for ax in axes:
            self.assertEqual(ax.get_yscale(), "log")
            self.assertEqual(ax.get_ylim(), axes[0].get_ylim())
        self.assertLess(axes[0].get_ylim()[0], 1)
        self.assertGreater(axes[0].get_ylim()[1], 4e5)
        lowstats = figures["lowstats.png"].axes[0]
        self.assertEqual(lowstats.get_legend_handles_labels()[1], list(frames))
        ticklabels = [text.get_text() for text in lowstats.get_xticklabels()]
        self.assertFalse(any("All empty" in text or "Unknown N" in text for text in ticklabels))
        self.assertEqual(plots._lowstat_rates(frames["model_0"], 2).loc["0", "recorded"], 0)

    def test_minimal_stage_and_six_model_comparison_write_nonempty_pngs(self):
        source = frame()
        history = {"train": [2., 1.], "validation": [3., 2.], "loss": "Loss", "best_epoch": 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plots.plot_stage(source, history, 2., root / "stage")
            self.assertEqual({p.name for p in (root / "stage").iterdir()},
                             {"test.png", "test_log.png", "distributions.png", "distributions_log.png", "loss.png"})
            frames = {f"model_{i}": source for i in range(6)}
            plots.plot_comparison(frames, {name: 2. for name in frames}, root / "comparison")
            self.assertEqual({p.name for p in (root / "comparison").iterdir()},
                             {"test.png", "test_log.png", "lowstats.png"})
            self.assertTrue(all(p.stat().st_size > 1000 for p in root.rglob("*.png")))



if __name__ == "__main__":
    unittest.main()
