"""Quick smoke-test for NLP variable scaling."""
import numpy as np
import top
from top.base import S_X, S_X_INV


def main():
    # Verify scaling constants
    print("S_X    :", S_X)
    print("S_X_INV:", S_X_INV)
    print("S_X * S_X_INV:", S_X * S_X_INV, "  (should be all 1.0)")

    # Quick test: create a Cruise optimizer and check that bounds are scaled
    opt = top.Cruise("A320", "KJFK", "EGLL", m0=0.85, perf_model="openap")
    opt.setup(nodes=20, debug=False)
    opt.init_conditions()

    # Check that x_guess is scaled
    xg = opt.x_guess
    print(f"\nx_guess shape: {xg.shape}")
    print(f"lat_s  range: [{xg[:,0].min():.3f}, {xg[:,0].max():.3f}]  (phys ~0.7-1.1 rad -> scaled ~7-11)")
    print(f"lon_s  range: [{xg[:,1].min():.3f}, {xg[:,1].max():.3f}]  (phys ~-1.2-0.0 rad -> scaled ~-12-0)")
    print(f"h_s    range: [{xg[:,2].min():.3f}, {xg[:,2].max():.3f}]  (phys ~6000-13000 m -> scaled ~0.6-1.3)")
    print(f"m_s    range: [{xg[:,3].min():.3f}, {xg[:,3].max():.3f}]  (phys ~50k-80k kg -> scaled ~0.7-1.1)")
    print(f"ts_s   range: [{xg[:,4].min():.1f}, {xg[:,4].max():.1f}]  (phys seconds, not scaled)")

    # Check bounds are scaled
    print(f"\nx_0_lb: {[f'{v:.4f}' for v in opt.x_0_lb]}")
    print(f"x_0_ub: {[f'{v:.4f}' for v in opt.x_0_ub]}")

    # Verify roundtrip: unscale should recover physical values
    xg_phys = xg * S_X_INV[np.newaxis, :]
    print(f"\nRoundtrip check (unscaled x_guess):")
    print(f"  lat (rad): [{xg_phys[:,0].min():.4f}, {xg_phys[:,0].max():.4f}]")
    print(f"  lon (rad): [{xg_phys[:,1].min():.4f}, {xg_phys[:,1].max():.4f}]")
    print(f"  h   (m)  : [{xg_phys[:,2].min():.0f}, {xg_phys[:,2].max():.0f}]")
    print(f"  m   (kg) : [{xg_phys[:,3].min():.0f}, {xg_phys[:,3].max():.0f}]")

    print("\n--- Running no-wind optimisation ---")
    try:
        df = opt.trajectory(objective="fuel")
        if df is not None:
            print(f"  SUCCESS: {len(df)} waypoints")
            print(f"  altitude: {df.altitude.min():.0f} - {df.altitude.max():.0f} ft")
            print(f"  mass:     {df.mass.min():.0f} - {df.mass.max():.0f} kg")
            print(f"  lat:      {df.latitude.min():.2f} - {df.latitude.max():.2f} deg")
            print(f"  lon:      {df.longitude.min():.2f} - {df.longitude.max():.2f} deg")
            print(f"  fuel:     {df.mass.iloc[0] - df.mass.iloc[-1]:.0f} kg")
        else:
            print("  FAILED: trajectory returned None")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
