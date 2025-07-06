from gwkokab import vts

filename = "pdet_with_TaylorF2Ecc_uniform_injections.hdf5"
filepath = "./" + filename


vts.train_regressor(
    data_path=filepath,
    input_keys=["mass_1_source","mass_2_source","a_1","a_2","eccentricity","redshift"],
    output_keys=["pdet"],
    width_size=256,
    depth=6,
    batch_size=1024,
    checkpoint_path="neural_" + filename,
    epochs=500,
    validation_split=0.1,
    learning_rate=1e-3,
)
